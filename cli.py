"""David vs Goliath — CLI with interactive wizard and flag-based commands.

Usage:
    dvg                        Interactive wizard
    dvg run  ...               Run a benchmark
    dvg demo ...               Quick single-question demo
    dvg distill ...            Distill Step-by-Step (LoRA fine-tuning)
    dvg cost ...               Cost estimation calculator
    dvg results                Show pre-computed results
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"
CACHE_DIR = APP_DIR / "cache"

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)

LOGO = r"""
[bold cyan]    ___            _    _                 ___       _ _       _   _
   |   \ __ ___ _(_)__| |  __ _____     / __|___  | (_) __ _| |_| |_
   | |) / _` \ V / / _` | \ V (_-<    | (_ / _ \ | | |/ _` |  _| ' \
   |___/\__,_|\_/|_\__,_|  \_//__/     \___\___/ |_|_|\__,_|\__|_||_|[/bold cyan]
"""
VERSION = "0.3.0"


def _banner():
    console.print(LOGO)
    console.print(f"  [dim]v{VERSION} — When tiny models outsmart the giants[/dim]\n")


def _short(name: str) -> str:
    return name.split("/")[-1].replace("-Instruct", "")


SLM_PRESETS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def _save_run_results(benchmark: str, slm: str, llm_model: str,
                      accuracies: dict, n_questions: int, seed: int):
    """Persist benchmark results to results/ as CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "run_history.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "benchmark", "slm_model", "llm_model",
                "n_questions", "seed", "strategy", "accuracy",
            ])
        ts = datetime.now().isoformat(timespec="seconds")
        for strat, acc in accuracies.items():
            writer.writerow([ts, benchmark, slm, llm_model, n_questions, seed, strat, f"{acc:.4f}"])
    console.print(f"  [dim]Results saved to {path}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive wizard
# ═══════════════════════════════════════════════════════════════════════════════


def _interactive():
    """Launch the guided interactive wizard."""
    from InquirerPy import inquirer

    _banner()

    action = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Showcase (highlight reel demo)", "value": "showcase"},
            {"name": "Run Benchmark", "value": "run"},
            {"name": "Quick Demo (single question)", "value": "demo"},
            {"name": "Distill (LoRA fine-tune with rationales)", "value": "distill"},
            {"name": "Cost Calculator", "value": "cost"},
            {"name": "View Pre-computed Results", "value": "results"},
            {"name": "Exit", "value": "exit"},
        ],
    ).execute()

    if action == "exit":
        raise typer.Exit()

    if action == "results":
        _cmd_results()
        return

    if action == "showcase":
        provider = inquirer.select("LLM provider:", choices=["openai", "anthropic"]).execute()
        default_llm = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"
        llm_model = inquirer.text("LLM model name:", default=default_llm).execute()
        import os
        env_key_sc = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
        api_key_sc = inquirer.secret("API key:", default=env_key_sc).execute()
        slm_sc = inquirer.select(
            "Select SLM model:",
            choices=[{"name": _short(s), "value": s} for s in SLM_PRESETS],
        ).execute()
        n_q = int(inquirer.number("Questions to evaluate:", default=30, min_allowed=10, max_allowed=100).execute())
        _cmd_showcase(slm=slm_sc, llm_provider=provider, llm_model=llm_model,
                      api_key=api_key_sc, n_questions=n_q)
        return

    if action == "cost":
        from pricing import LLM_PRICING
        models = list(LLM_PRICING.keys())
        direct = inquirer.select("Your current LLM:", choices=models).execute()
        selector = inquirer.select("Selector LLM for hybrid:", choices=models, default="gpt-4o-mini").execute()
        vol = int(inquirer.number("Monthly request volume:", default=100000, min_allowed=100).execute())
        din = int(inquirer.number("Avg input tokens per request (your LLM):", default=500).execute())
        dout = int(inquirer.number("Avg output tokens per request (your LLM):", default=50).execute())
        dacc = float(inquirer.number("Your LLM's accuracy (0.0-1.0):", default=0.90).execute())
        _cmd_cost(
            direct_model=direct, selector_model=selector,
            monthly_volume=vol, direct_input_tpq=din,
            direct_output_tpq=dout, direct_accuracy=dacc,
        )
        return

    # For run, demo, and distill — gather model config
    slm = inquirer.select(
        "Select SLM model:",
        choices=[{"name": _short(s), "value": s} for s in SLM_PRESETS]
        + [{"name": "Custom HuggingFace ID", "value": "__custom__"}],
    ).execute()
    if slm == "__custom__":
        slm = inquirer.text("Enter HuggingFace model ID:").execute()

    provider = inquirer.select("LLM provider:", choices=["openai", "anthropic"]).execute()
    default_llm = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"
    llm_model = inquirer.text("LLM model name:", default=default_llm).execute()

    import os
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    api_key = inquirer.secret("API key:", default=env_key).execute()

    if action == "demo":
        _cmd_demo(slm=slm, llm_provider=provider, llm_model=llm_model, api_key=api_key)
        return

    from tasks import BENCHMARK_REGISTRY
    bench = inquirer.select(
        "Benchmark:",
        choices=[{"name": v["label"], "value": k} for k, v in BENCHMARK_REGISTRY.items()],
    ).execute()

    if action == "distill":
        n_train = int(inquirer.number("Training examples:", default=30, min_allowed=10, max_allowed=100).execute())
        n_test = int(inquirer.number("Test examples:", default=20, min_allowed=5, max_allowed=100).execute())
        lam = float(inquirer.number("Lambda (label weight, 0.0-1.0):", default=0.5).execute())
        epochs = int(inquirer.number("Fine-tuning epochs:", default=3, min_allowed=1, max_allowed=10).execute())
        seed = int(inquirer.number("Seed:", default=42).execute())
        _cmd_distill(
            slm=slm, benchmark=bench, llm_provider=provider, llm_model=llm_model,
            api_key=api_key, n_train=n_train, n_test=n_test,
            lambda_weight=lam, epochs=epochs, seed=seed,
        )
        return

    # action == "run"
    shots = int(inquirer.number("Few-shot examples (k):", default=3, min_allowed=1, max_allowed=10).execute())
    sample = int(inquirer.number("Test questions:", default=20, min_allowed=5, max_allowed=200).execute())
    seed = int(inquirer.number("Seed:", default=42).execute())

    _cmd_run(
        slm=slm, benchmark=bench, llm_provider=provider, llm_model=llm_model,
        api_key=api_key, shots=shots, sample_size=sample, seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════════════════


@app.callback()
def main(ctx: typer.Context):
    """David vs Goliath — When tiny models outsmart the giants."""
    if ctx.invoked_subcommand is None:
        _interactive()


@app.command()
def run(
    slm: str = typer.Option("HuggingFaceTB/SmolLM2-135M-Instruct", help="HuggingFace SLM model ID"),
    benchmark: str = typer.Option("boolq", help="Benchmark name: boolq, sst2, agnews"),
    llm_provider: str = typer.Option("openai", help="LLM provider: openai, anthropic"),
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    api_key: str = typer.Option("", envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"], help="LLM API key"),
    shots: int = typer.Option(3, help="Number of few-shot examples"),
    sample_size: int = typer.Option(20, help="Number of test questions"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Run a full benchmark evaluation."""
    _cmd_run(slm=slm, benchmark=benchmark, llm_provider=llm_provider,
             llm_model=llm_model, api_key=api_key, shots=shots,
             sample_size=sample_size, seed=seed)


def _cmd_run(*, slm, benchmark, llm_provider, llm_model, api_key, shots, sample_size, seed):
    _banner()

    from tasks import load_task
    from slm import load_model, generate_answer_generic
    from selector import RandomSelector, GenericLLMSelector
    from pricing import CostEstimate, project_monthly

    console.print(Panel(
        f"[bold]SLM:[/bold] {_short(slm)}  |  "
        f"[bold]Benchmark:[/bold] {benchmark}  |  "
        f"[bold]LLM:[/bold] {llm_model}\n"
        f"[bold]Shots:[/bold] {shots}  |  "
        f"[bold]Samples:[/bold] {sample_size}  |  "
        f"[bold]Seed:[/bold] {seed}",
        title="Configuration", border_style="cyan",
    ))

    with console.status("[bold green]Loading benchmark dataset..."):
        task = load_task(benchmark, n_train=50, n_test=sample_size, seed=seed)
    console.print(f"  {task.name}: {len(task.train_pool)} train, {len(task.test_set)} test\n")

    with console.status(f"[bold green]Loading {_short(slm)}..."):
        model, tokenizer, device = load_model(slm)
    console.print(f"  Model loaded on {device}\n")

    strategies = ["Zero-Shot", "Random"]
    llm_sel = None
    if api_key:
        strategies.append("LLM-Assisted")
        llm_sel = GenericLLMSelector(
            llm_provider, llm_model, api_key,
            task_instruction=task.instruction, task_choices=task.choices,
            cache_dir=CACHE_DIR / "selector",
        )

    random_sel = RandomSelector()
    results = {s: [] for s in strategies}
    total_slm_tokens = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Evaluating...", total=len(task.test_set))

        for i, test_ex in enumerate(task.test_set):
            # Zero-shot
            ans, _, tok = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, [],
            )
            results["Zero-Shot"].append(ans == test_ex.correct)
            total_slm_tokens += tok

            # Random
            r_shots = random_sel.select(task.train_pool, test_ex, shots, seed=seed + i)
            ans, _, tok = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, r_shots,
            )
            results["Random"].append(ans == test_ex.correct)
            total_slm_tokens += tok

            # LLM-Assisted
            if llm_sel:
                l_shots = llm_sel.select(task.train_pool, test_ex, shots, seed=seed + i)
                ans, _, tok = generate_answer_generic(
                    model, tokenizer, device,
                    task.instruction, task.choices,
                    test_ex.input_text, test_ex.context, l_shots,
                )
                results["LLM-Assisted"].append(ans == test_ex.correct)
                total_slm_tokens += tok

            progress.update(ptask, advance=1)

    # Results table
    accuracies = {s: sum(r) / len(r) if r else 0.0 for s, r in results.items()}

    table = Table(title=f"\n{task.name} Results ({len(task.test_set)} questions)",
                  box=box.ROUNDED, border_style="green")
    table.add_column("Strategy", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct", justify="right")

    for strat, acc in accuracies.items():
        n_correct = sum(results[strat])
        color = "green" if acc >= max(accuracies.values()) else "white"
        table.add_row(strat, f"[{color}]{acc:.1%}[/{color}]", f"{n_correct}/{len(results[strat])}")
    console.print(table)

    # Persist results
    _save_run_results(benchmark, slm, llm_model, accuracies, len(task.test_set), seed)

    # Cost estimation (token-based, from measured data)
    if llm_sel:
        cost_est = CostEstimate(
            llm_input_tokens=llm_sel.total_input_tokens,
            llm_output_tokens=llm_sel.total_output_tokens,
            slm_input_tokens=total_slm_tokens,
            n_questions=len(task.test_set),
            accuracy_zero_shot=accuracies.get("Zero-Shot", 0),
            accuracy_random=accuracies.get("Random", 0),
            accuracy_llm_assisted=accuracies.get("LLM-Assisted", 0),
        )
        proj = project_monthly(
            100_000,
            direct_model=llm_model,
            selector_model=llm_model,
            measured=cost_est,
        )

        console.print(f"\n  [dim]Measured: {cost_est.avg_llm_input_per_q:.0f} selector input tokens/q, "
                      f"{cost_est.avg_llm_output_per_q:.0f} output tokens/q[/dim]")

        cost_table = Table(title="\nCost Projection (100K calls/month, self-hosted SLM)",
                           box=box.ROUNDED, border_style="yellow")
        cost_table.add_column("Approach", style="bold")
        cost_table.add_column("Monthly Cost", justify="right")
        cost_table.add_column("Accuracy", justify="right")

        cost_table.add_row("Pure LLM", f"${proj.pure_llm_cost:,.2f}", f"{proj.direct_accuracy:.0%}")
        cost_table.add_row("Hybrid (LLM-Assisted)", f"${proj.hybrid_total:,.2f}",
                           f"{cost_est.accuracy_llm_assisted:.1%}")
        cost_table.add_row(
            "[bold green]Savings[/bold green]",
            f"[bold green]${proj.savings_abs:,.2f}/mo ({proj.savings_pct:.0f}%)[/bold green]",
            "",
        )
        console.print(cost_table)

    console.print("\n[dim]Done.[/dim]")


@app.command()
def demo(
    slm: str = typer.Option("HuggingFaceTB/SmolLM2-135M-Instruct", help="HuggingFace SLM model ID"),
    llm_provider: str = typer.Option("openai", help="LLM provider: openai, anthropic"),
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    api_key: str = typer.Option("", envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"], help="LLM API key"),
    shots: int = typer.Option(3, help="Number of few-shot examples"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Quick demo: run a single random BoolQ question through all strategies."""
    _cmd_demo(slm=slm, llm_provider=llm_provider, llm_model=llm_model,
              api_key=api_key, shots=shots, seed=seed)


def _cmd_demo(*, slm, llm_provider, llm_model, api_key, shots=3, seed=42):
    import random as stdlib_random
    _banner()

    from tasks import load_boolq_task
    from slm import load_model, generate_answer_generic
    from selector import RandomSelector, GenericLLMSelector

    with console.status("[bold green]Loading model & data..."):
        model, tokenizer, device = load_model(slm)
        task = load_boolq_task(n_train=50, n_test=20, seed=seed)

    test_ex = stdlib_random.choice(task.test_set)

    console.print(Panel(
        f"[bold]Q:[/bold] {test_ex.input_text}\n\n"
        f"[dim]{test_ex.context[:200]}...[/dim]\n\n"
        f"[bold]Ground truth:[/bold] {test_ex.correct}",
        title="Random BoolQ Question", border_style="blue",
    ))

    panels = []

    # Zero-shot
    with console.status("Zero-shot..."):
        ans_zs, raw_zs, _ = generate_answer_generic(
            model, tokenizer, device,
            task.instruction, task.choices,
            test_ex.input_text, test_ex.context, [],
        )
    mark = "+" if ans_zs == test_ex.correct else "x"
    panels.append(Panel(f"{mark} [bold]{ans_zs}[/bold]\n[dim]Raw: {raw_zs.strip()[:50]}[/dim]",
                        title="Zero-Shot", border_style="white"))

    # Random
    with console.status("Random selection..."):
        r_shots = RandomSelector().select(task.train_pool, test_ex, shots, seed=seed)
        ans_r, raw_r, _ = generate_answer_generic(
            model, tokenizer, device,
            task.instruction, task.choices,
            test_ex.input_text, test_ex.context, r_shots,
        )
    mark = "+" if ans_r == test_ex.correct else "x"
    panels.append(Panel(f"{mark} [bold]{ans_r}[/bold]\n[dim]{shots} random examples[/dim]",
                        title="Random", border_style="yellow"))

    # LLM-Assisted
    if api_key:
        with console.status("LLM selecting examples..."):
            llm_sel = GenericLLMSelector(
                llm_provider, llm_model, api_key,
                task_instruction=task.instruction, task_choices=task.choices,
                cache_dir=CACHE_DIR / "selector",
            )
            l_shots = llm_sel.select(task.train_pool, test_ex, shots, seed=seed)
        with console.status("LLM-Assisted inference..."):
            ans_l, raw_l, _ = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, l_shots,
            )
        mark = "+" if ans_l == test_ex.correct else "x"
        panels.append(Panel(f"{mark} [bold]{ans_l}[/bold]\n[dim]{shots} LLM-picked examples[/dim]",
                            title="LLM-Assisted", border_style="green"))
    else:
        panels.append(Panel("[dim]No API key — skipped[/dim]",
                            title="LLM-Assisted", border_style="red"))

    console.print()
    from rich.columns import Columns
    console.print(Columns(panels, equal=True, expand=True))
    console.print()


@app.command()
def showcase(
    slm: str = typer.Option("HuggingFaceTB/SmolLM2-360M-Instruct", help="HuggingFace SLM model ID"),
    benchmark: str = typer.Option("boolq", help="Benchmark: boolq, sst2, agnews"),
    llm_provider: str = typer.Option("openai", help="LLM provider: openai, anthropic"),
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    api_key: str = typer.Option("", envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"], help="LLM API key"),
    n_questions: int = typer.Option(30, help="Questions to evaluate (more = better highlights)"),
    shots: int = typer.Option(3, help="Few-shot examples per question"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Showcase: run a batch and present the best highlight-reel moments."""
    _cmd_showcase(slm=slm, benchmark=benchmark, llm_provider=llm_provider,
                  llm_model=llm_model, api_key=api_key, n_questions=n_questions,
                  shots=shots, seed=seed)


def _cmd_showcase(*, slm, benchmark="boolq", llm_provider, llm_model, api_key,
                  n_questions=30, shots=3, seed=42):
    _banner()

    if not api_key:
        console.print("[red]Showcase requires an API key for the LLM selector.[/red]")
        raise typer.Exit(1)

    from tasks import load_task
    from slm import load_model, generate_answer_generic
    from selector import RandomSelector, GenericLLMSelector
    from pricing import get_model_price

    console.print(Panel(
        f"[bold]SLM:[/bold] {_short(slm)}  |  [bold]LLM:[/bold] {llm_model}  |  "
        f"[bold]Benchmark:[/bold] {benchmark}\n"
        f"[bold]Questions:[/bold] {n_questions}  |  [bold]Shots:[/bold] {shots}  |  "
        f"[bold]Seed:[/bold] {seed}",
        title="Showcase Configuration", border_style="cyan",
    ))

    # Load everything
    with console.status("[bold green]Loading benchmark dataset..."):
        task = load_task(benchmark, n_train=50, n_test=n_questions, seed=seed)
    console.print(f"  {task.name}: {len(task.train_pool)} train, {len(task.test_set)} test")

    with console.status(f"[bold green]Loading {_short(slm)}..."):
        model, tokenizer, device = load_model(slm)
    console.print(f"  Model loaded on {device}\n")

    llm_sel = GenericLLMSelector(
        llm_provider, llm_model, api_key,
        task_instruction=task.instruction, task_choices=task.choices,
        cache_dir=CACHE_DIR / "selector",
    )
    random_sel = RandomSelector()

    # Evaluate all questions, track per-question results
    question_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Running showcase evaluation...", total=len(task.test_set))

        for i, test_ex in enumerate(task.test_set):
            qr = {"example": test_ex, "shots_used": None}

            # Zero-shot
            ans_zs, raw_zs, _ = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, [],
            )
            qr["zs"] = (ans_zs, raw_zs, ans_zs == test_ex.correct)

            # Random
            r_shots = random_sel.select(task.train_pool, test_ex, shots, seed=seed + i)
            ans_r, raw_r, _ = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, r_shots,
            )
            qr["rand"] = (ans_r, raw_r, ans_r == test_ex.correct)

            # LLM-Assisted
            l_shots = llm_sel.select(task.train_pool, test_ex, shots, seed=seed + i)
            ans_l, raw_l, _ = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, l_shots,
            )
            qr["llm"] = (ans_l, raw_l, ans_l == test_ex.correct)
            qr["shots_used"] = l_shots

            question_results.append(qr)
            progress.update(ptask, advance=1)

    # ── Find highlights: LLM-Assisted correct, others wrong ──────────────
    highlights = [
        qr for qr in question_results
        if qr["llm"][2] and (not qr["zs"][2] or not qr["rand"][2])
    ]
    # Sort: prefer cases where BOTH zero-shot AND random failed
    highlights.sort(key=lambda qr: (not qr["zs"][2]) + (not qr["rand"][2]), reverse=True)
    highlights = highlights[:4]  # Show up to 4

    console.print()

    if highlights:
        console.print(Panel(
            "[bold]These questions stumped the small model — until the LLM picked the right examples.[/bold]",
            title="The David vs Goliath Effect",
            border_style="bold cyan",
        ))

        for idx, qr in enumerate(highlights, 1):
            ex = qr["example"]
            zs_ans, _, zs_ok = qr["zs"]
            r_ans, _, r_ok = qr["rand"]
            l_ans, _, l_ok = qr["llm"]

            zs_mark = "[green]Correct[/green]" if zs_ok else "[red]Wrong[/red]"
            r_mark = "[green]Correct[/green]" if r_ok else "[red]Wrong[/red]"
            l_mark = "[bold green]Correct[/bold green]"

            # Build the highlight panel
            lines = []
            lines.append(f"[bold]Q:[/bold] {ex.input_text}")
            if ex.context:
                ctx_preview = ex.context[:150] + ("..." if len(ex.context) > 150 else "")
                lines.append(f"[dim]{ctx_preview}[/dim]")
            lines.append(f"[bold]Answer:[/bold] {ex.correct}")
            lines.append("")
            lines.append(f"  Zero-Shot:    {zs_ans:>10}  {zs_mark}")
            lines.append(f"  Random:       {r_ans:>10}  {r_mark}")
            lines.append(f"  LLM-Assisted: {l_ans:>10}  {l_mark}")

            # Show what examples the LLM picked
            if qr["shots_used"]:
                lines.append("")
                lines.append("[bold]Examples the LLM chose:[/bold]")
                for j, shot in enumerate(qr["shots_used"], 1):
                    shot_text = shot.input_text[:70] + ("..." if len(shot.input_text) > 70 else "")
                    lines.append(f"  {j}. \"{shot_text}\" -> [bold]{shot.correct}[/bold]")

            console.print(Panel(
                "\n".join(lines),
                title=f"Highlight {idx}",
                border_style="green",
            ))
    else:
        console.print("[yellow]No clear highlights found where LLM-Assisted uniquely won. "
                      "Try with more questions (--n-questions 50).[/yellow]")

    # ── Aggregate results ────────────────────────────────────────────────
    n = len(question_results)
    zs_acc = sum(1 for qr in question_results if qr["zs"][2]) / n
    r_acc = sum(1 for qr in question_results if qr["rand"][2]) / n
    l_acc = sum(1 for qr in question_results if qr["llm"][2]) / n
    boost = l_acc - zs_acc

    summary_table = Table(box=box.ROUNDED, border_style="cyan", title=f"\nResults — {n} questions")
    summary_table.add_column("Strategy", style="bold")
    summary_table.add_column("Accuracy", justify="right")
    summary_table.add_column("Correct", justify="right")

    for label, acc, ok_count in [
        ("Zero-Shot", zs_acc, sum(1 for qr in question_results if qr["zs"][2])),
        ("Random Few-Shot", r_acc, sum(1 for qr in question_results if qr["rand"][2])),
        ("LLM-Assisted", l_acc, sum(1 for qr in question_results if qr["llm"][2])),
    ]:
        color = "green" if acc == max(zs_acc, r_acc, l_acc) else "white"
        summary_table.add_row(label, f"[{color}]{acc:.1%}[/{color}]", f"{ok_count}/{n}")
    console.print(summary_table)

    # ── Cost breakdown ───────────────────────────────────────────────────
    selector_price = get_model_price(llm_model)
    total_selector_cost = (
        llm_sel.total_input_tokens * selector_price["input"] / 1e6
        + llm_sel.total_output_tokens * selector_price["output"] / 1e6
    )
    cost_per_q = total_selector_cost / max(n, 1)

    # What would pure LLM cost for the same questions?
    direct_price = get_model_price("gpt-4o")
    direct_cost = n * (350 * direct_price["input"] / 1e6 + 20 * direct_price["output"] / 1e6)

    console.print(Panel(
        f"[bold]LLM selector cost:[/bold] ${total_selector_cost:.4f} "
        f"(${cost_per_q:.5f}/question)\n"
        f"[bold]Pure GPT-4o equivalent:[/bold] ${direct_cost:.4f} "
        f"(${direct_cost/max(n,1):.5f}/question)\n"
        f"[bold]Cost reduction:[/bold] [green]{(1 - total_selector_cost/max(direct_cost, 0.0001))*100:.0f}%[/green]\n\n"
        f"[bold]Accuracy boost:[/bold] [green]+{boost*100:.1f} percentage points[/green] over zero-shot\n"
        f"[bold]Highlights:[/bold] {len(highlights)} questions where LLM-Assisted uniquely won\n\n"
        f"[dim]{llm_sel.total_input_tokens:,} input + {llm_sel.total_output_tokens:,} output tokens used[/dim]",
        title="The Bottom Line",
        border_style="bold yellow",
    ))

    # Persist results
    accuracies = {"Zero-Shot": zs_acc, "Random": r_acc, "LLM-Assisted": l_acc}
    _save_run_results(benchmark, slm, llm_model, accuracies, n, seed)

    console.print()


@app.command()
def cost(
    direct_model: str = typer.Option("gpt-4o-mini", help="Your current LLM model"),
    selector_model: str = typer.Option("gpt-4o-mini", help="Selector LLM for hybrid"),
    monthly_volume: int = typer.Option(100_000, help="Monthly request volume"),
    direct_input_tpq: int = typer.Option(500, help="Avg input tokens per request (your LLM)"),
    direct_output_tpq: int = typer.Option(50, help="Avg output tokens per request (your LLM)"),
    direct_accuracy: float = typer.Option(0.90, help="Your LLM's accuracy (0.0-1.0)"),
):
    """Estimate your savings switching from pure LLM to hybrid."""
    _cmd_cost(
        direct_model=direct_model, selector_model=selector_model,
        monthly_volume=monthly_volume,
        direct_input_tpq=direct_input_tpq, direct_output_tpq=direct_output_tpq,
        direct_accuracy=direct_accuracy,
    )


def _cmd_cost(*, direct_model, selector_model, monthly_volume,
              direct_input_tpq, direct_output_tpq, direct_accuracy):
    _banner()
    from pricing import quick_cost_table, LLM_PRICING, project_monthly

    if direct_model not in LLM_PRICING:
        console.print(f"[yellow]Unknown model '{direct_model}', using gpt-4o-mini pricing.[/yellow]")
        direct_model = "gpt-4o-mini"

    console.print(Panel(
        f"[bold]Your LLM:[/bold] {direct_model}  "
        f"({direct_input_tpq} in + {direct_output_tpq} out tokens/question)\n"
        f"[bold]Selector:[/bold] {selector_model}  |  "
        f"[bold]Volume:[/bold] {monthly_volume:,}/mo  |  "
        f"[bold]Your accuracy:[/bold] {direct_accuracy:.0%}",
        title="Your Savings Estimate", border_style="yellow",
    ))

    rows = quick_cost_table(
        direct_model, selector_model, monthly_volume,
        direct_input_tpq=float(direct_input_tpq),
        direct_output_tpq=float(direct_output_tpq),
        direct_accuracy=direct_accuracy,
    )

    table = Table(
        title=f"\nCost Comparison — {monthly_volume:,} calls/month",
        box=box.ROUNDED, border_style="yellow",
    )
    table.add_column("SLM Hosting", style="bold")
    table.add_column("Pure LLM", justify="right")
    table.add_column("Hybrid", justify="right")
    table.add_column("Savings", justify="right", style="green")
    table.add_column("Break-even", justify="right", style="dim")

    for r in rows:
        be = f"{r['break_even']:,}" if r["break_even"] > 0 else "—"
        table.add_row(
            r["hosting"],
            f"${r['pure_llm_cost']:,.2f}",
            f"${r['hybrid_total']:,.2f}",
            f"${r['savings_abs']:,.2f} ({r['savings_pct']:.0f}%)",
            be,
        )
    console.print(table)

    proj = project_monthly(
        monthly_volume,
        direct_model=direct_model,
        direct_input_tpq=float(direct_input_tpq),
        direct_output_tpq=float(direct_output_tpq),
        selector_model=selector_model,
    )
    if proj.savings_abs > 0:
        console.print(f"\n  [bold green]Annual savings: ${proj.savings_abs * 12:,.0f}[/bold green] "
                      f"(with self-hosted SLM)\n")
    console.print()


@app.command()
def distill(
    slm: str = typer.Option("HuggingFaceTB/SmolLM2-135M-Instruct", help="Student SLM"),
    benchmark: str = typer.Option("boolq", help="Benchmark: boolq, sst2, agnews"),
    llm_provider: str = typer.Option("openai", help="Teacher LLM provider"),
    llm_model: str = typer.Option("gpt-4o-mini", help="Teacher LLM model"),
    api_key: str = typer.Option("", envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
    n_train: int = typer.Option(30, help="Training examples"),
    n_test: int = typer.Option(20, help="Test examples"),
    lambda_weight: float = typer.Option(0.5, help="Label weight (0.0-1.0)"),
    epochs: int = typer.Option(3, help="LoRA fine-tuning epochs"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Distill Step-by-Step: teacher rationales + LoRA fine-tuning."""
    _cmd_distill(
        slm=slm, benchmark=benchmark, llm_provider=llm_provider,
        llm_model=llm_model, api_key=api_key, n_train=n_train, n_test=n_test,
        lambda_weight=lambda_weight, epochs=epochs, seed=seed,
    )


def _cmd_distill(*, slm, benchmark, llm_provider, llm_model, api_key,
                 n_train, n_test, lambda_weight, epochs, seed):
    _banner()

    from tasks import load_task
    from distill import (
        generate_teacher_rationales,
        DistillDataset,
        fine_tune_student,
        evaluate_distilled,
    )

    console.print(Panel(
        f"[bold]Student SLM:[/bold] {_short(slm)}\n"
        f"[bold]Teacher LLM:[/bold] {llm_model} ({llm_provider})\n"
        f"[bold]Benchmark:[/bold] {benchmark}  |  "
        f"[bold]Train:[/bold] {n_train}  |  [bold]Test:[/bold] {n_test}\n"
        f"[bold]Lambda:[/bold] {lambda_weight}  |  "
        f"[bold]Epochs:[/bold] {epochs}  |  [bold]Seed:[/bold] {seed}",
        title="Distillation Config", border_style="magenta",
    ))

    console.print("\n[bold magenta]Phase 1[/bold magenta] — Loading benchmark")
    with console.status("[bold green]Loading dataset..."):
        task = load_task(benchmark, n_train=n_train, n_test=n_test, seed=seed)
    console.print(f"  {task.name}: {len(task.train_pool)} train, {len(task.test_set)} test\n")

    console.print("[bold magenta]Phase 2[/bold magenta] — Teacher rationale generation")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Generating rationales...", total=len(task.train_pool))

        def _rat_cb(i, _t):
            progress.update(ptask, completed=i)

        annotated = generate_teacher_rationales(
            task, llm_provider, llm_model, api_key, callback=_rat_cb,
        )

    console.print(f"  {len(annotated)} examples annotated\n")

    sample_table = Table(title="Sample Rationales", box=box.SIMPLE, border_style="dim")
    sample_table.add_column("Question (truncated)", max_width=50)
    sample_table.add_column("Label", style="bold")
    sample_table.add_column("Rationale", max_width=60)
    for ex in annotated[:3]:
        sample_table.add_row(
            ex.input_text[:50] + "...",
            ex.label,
            ex.rationale[:60] + ("..." if len(ex.rationale) > 60 else ""),
        )
    console.print(sample_table)
    console.print()

    console.print("[bold magenta]Phase 3[/bold magenta] — Baseline evaluation (original student)")
    with console.status(f"[bold green]Loading {_short(slm)}..."):
        from slm import load_model
        model_orig, tok_orig, dev_orig = load_model(slm)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Evaluating baseline...", total=len(task.test_set))
        def _base_cb(i, _t): progress.update(ptask, completed=i)
        baseline = evaluate_distilled(model_orig, tok_orig, dev_orig, task, callback=_base_cb)

    console.print(f"  Baseline accuracy: [bold]{baseline['accuracy']:.1%}[/bold] "
                  f"({baseline['n_correct']}/{baseline['n_total']})\n")

    # Free the original model — fine_tune_student loads a fresh copy
    del model_orig, tok_orig
    import gc; gc.collect()

    console.print("[bold magenta]Phase 4[/bold magenta] — Multi-task dataset + LoRA fine-tuning")
    console.print(f"  Loss: L = {lambda_weight} * L_label + {1-lambda_weight:.2f} * L_rationale")

    train_annotated = annotated[:n_train]

    # Use the tokenizer from fine_tune_student's model load to avoid an extra load
    with console.status("[bold green]Preparing dataset & loading model..."):
        # Load model once — reuse tokenizer for dataset
        from slm import load_model as _lm
        _ft_model, _ft_tok, _ft_dev = _lm(slm)
        dataset = DistillDataset(
            train_annotated, task.instruction, task.choices,
            _ft_tok, max_len=512, lambda_weight=lambda_weight, seed=seed,
        )
    console.print(f"  Dataset: {len(dataset)} examples\n")

    # Free — fine_tune_student will load its own copy for LoRA wrapping
    del _ft_model, _ft_tok
    gc.collect()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Fine-tuning...", total=1)

        def _train_cb(step, total_steps, loss_val):
            progress.update(ptask, total=total_steps, completed=step,
                            description=f"Step {step}/{total_steps}  loss={loss_val:.4f}")

        model_d, tok_d, dev_d = fine_tune_student(
            slm, dataset, epochs=epochs, lr=2e-4, callback=_train_cb,
        )

    console.print()
    console.print("[bold magenta]Phase 5[/bold magenta] — Evaluate distilled student")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Evaluating distilled...", total=len(task.test_set))
        def _dist_cb(i, _t): progress.update(ptask, completed=i)
        distilled = evaluate_distilled(model_d, tok_d, dev_d, task, callback=_dist_cb)

    delta = distilled["accuracy"] - baseline["accuracy"]

    result_table = Table(
        title=f"\nDistillation Results — {task.name}",
        box=box.ROUNDED, border_style="magenta",
    )
    result_table.add_column("Model", style="bold")
    result_table.add_column("Accuracy", justify="right")
    result_table.add_column("Correct", justify="right")
    result_table.add_column("Delta", justify="right")

    result_table.add_row(
        f"{_short(slm)} (original)",
        f"{baseline['accuracy']:.1%}",
        f"{baseline['n_correct']}/{baseline['n_total']}",
        "",
    )
    delta_color = "green" if delta > 0 else ("red" if delta < 0 else "white")
    result_table.add_row(
        f"{_short(slm)} (distilled)",
        f"[bold {delta_color}]{distilled['accuracy']:.1%}[/bold {delta_color}]",
        f"{distilled['n_correct']}/{distilled['n_total']}",
        f"[bold {delta_color}]{delta:+.1%}[/bold {delta_color}]",
    )
    console.print(result_table)

    console.print(Panel(
        f"[bold]Multi-task loss:[/bold] L = {lambda_weight} * L_label + "
        f"{1-lambda_weight:.2f} * L_rationale\n"
        f"[bold]Method:[/bold] Deterministic stratified dataset mixing\n"
        f"[bold]Adaptation:[/bold] LoRA (rank {8}, alpha {16}) — ~1-3% of parameters trainable\n"
        f"[bold]Teacher:[/bold] {llm_model} -> rationales + labels\n"
        f"[bold]Student:[/bold] {_short(slm)} -> trained to predict labels AND generate rationales",
        title="Loss Function Details", border_style="dim",
    ))
    console.print()


@app.command()
def results():
    """Display pre-computed benchmark results from CSV data."""
    _cmd_results()


def _cmd_results():
    _banner()
    import pandas as pd

    csv_path = DATA_DIR / "all_results_summary.csv"
    if not csv_path.exists():
        console.print("[red]No results found.[/red] Run a benchmark first or check data/ folder.")
        raise typer.Exit(1)

    df = pd.read_csv(csv_path)

    table = Table(title="\nPre-computed Benchmark Results", box=box.ROUNDED, border_style="cyan")
    table.add_column("SLM Model", style="bold")
    table.add_column("Strategy")
    table.add_column("LLM Selector")
    table.add_column("Seed", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Std", justify="right")

    for _, row in df.iterrows():
        short = _short(row["slm_model"])
        acc = row["mean_accuracy"]
        best = acc == df[df["slm_model"] == row["slm_model"]]["mean_accuracy"].max()
        color = "green" if best else "white"
        table.add_row(
            short,
            row["strategy"],
            str(row["llm_selector"]),
            str(row["seed"]),
            f"[{color}]{acc:.1%}[/{color}]",
            f"{row['std']:.4f}",
        )

    console.print(table)
    console.print(f"\n[dim]{len(df)} configurations across "
                  f"{df['seed'].nunique()} seeds.[/dim]\n")

    # Also show run history if available
    history_path = RESULTS_DIR / "run_history.csv"
    if history_path.exists():
        hdf = pd.read_csv(history_path)
        console.print(f"[dim]+ {len(hdf)} rows in run history ({history_path})[/dim]\n")


if __name__ == "__main__":
    app()
