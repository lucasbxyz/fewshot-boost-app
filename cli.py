"""FewShot Boost — CLI with interactive wizard and flag-based commands.

Usage:
    fewshot-boost              Interactive wizard
    fewshot-boost run  ...     Run a benchmark
    fewshot-boost demo ...     Quick single-question demo
    fewshot-boost distill ...  Distill Step-by-Step (LoRA fine-tuning)
    fewshot-boost cost ...     Cost estimation calculator
    fewshot-boost results      Show pre-computed results
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)

LOGO = r"""
[bold cyan]    ___              __ _        _     ___             _
   | __|___ __ __ __/ _| |_  ___| |_  | _ ) ___  ___ | |_
   | _/ -_) V  V /\__ \ ' \/ _ \  _| | _ \/ _ \/ _ \|  _|
   |_|\___|\_/\_/ |___/_||_\___/\__| |___/\___/\___/ \__|[/bold cyan]
"""
VERSION = "0.1.0"


def _banner():
    console.print(LOGO)
    console.print(f"  [dim]v{VERSION} — Intelligent few-shot selection for small language models[/dim]\n")


def _short(name: str) -> str:
    return name.split("/")[-1].replace("-Instruct", "")


SLM_PRESETS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


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
            {"name": "🚀 Run Benchmark", "value": "run"},
            {"name": "⚡ Quick Demo (single question)", "value": "demo"},
            {"name": "🎓 Distill (LoRA fine-tune with rationales)", "value": "distill"},
            {"name": "💰 Cost Calculator", "value": "cost"},
            {"name": "📊 View Pre-computed Results", "value": "results"},
            {"name": "👋 Exit", "value": "exit"},
        ],
    ).execute()

    if action == "exit":
        raise typer.Exit()

    if action == "results":
        _cmd_results()
        return

    if action == "cost":
        model = inquirer.select("LLM model for cost estimation:", choices=list(
            __import__("pricing").LLM_PRICING.keys()
        )).execute()
        vol = int(inquirer.number("Monthly API call volume:", default=100000, min_allowed=100).execute())
        _cmd_cost(llm_model=model, monthly_volume=vol)
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
    """FewShot Boost — Make tiny models smarter with intelligent few-shot selection."""
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
    from pricing import CostEstimate

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
    console.print(f"  ✓ {task.name}: {len(task.train_pool)} train, {len(task.test_set)} test\n")

    with console.status(f"[bold green]Loading {_short(slm)}..."):
        model, tokenizer, device = load_model(slm)
    console.print(f"  ✓ Model loaded on {device}\n")

    strategies = ["Zero-Shot", "Random"]
    llm_sel = None
    if api_key:
        strategies.append("LLM-Assisted")
        llm_sel = GenericLLMSelector(
            llm_provider, llm_model, api_key,
            task_instruction=task.instruction, task_choices=task.choices,
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

    # Cost estimation
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
        proj = cost_est.project_monthly(100_000, llm_model, "self_hosted_cpu")

        cost_table = Table(title="\nCost Projection (100K calls/month, self-hosted SLM)",
                           box=box.ROUNDED, border_style="yellow")
        cost_table.add_column("Approach", style="bold")
        cost_table.add_column("Monthly Cost", justify="right")
        cost_table.add_column("Accuracy", justify="right")

        cost_table.add_row("Pure LLM", f"${proj['pure_llm_cost']:,.2f}", "~90%")
        cost_table.add_row("Hybrid (LLM-Assisted)", f"${proj['hybrid_total']:,.2f}",
                           f"{cost_est.accuracy_llm_assisted:.1%}")
        cost_table.add_row(
            "[bold green]Savings[/bold green]",
            f"[bold green]${proj['savings_abs']:,.2f}/mo ({proj['savings_pct']:.0f}%)[/bold green]",
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
    mark = "✅" if ans_zs == test_ex.correct else "❌"
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
    mark = "✅" if ans_r == test_ex.correct else "❌"
    panels.append(Panel(f"{mark} [bold]{ans_r}[/bold]\n[dim]{shots} random examples[/dim]",
                        title="Random", border_style="yellow"))

    # LLM-Assisted
    if api_key:
        with console.status("LLM selecting examples..."):
            llm_sel = GenericLLMSelector(
                llm_provider, llm_model, api_key,
                task_instruction=task.instruction, task_choices=task.choices,
            )
            l_shots = llm_sel.select(task.train_pool, test_ex, shots, seed=seed)
        with console.status("LLM-Assisted inference..."):
            ans_l, raw_l, _ = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, l_shots,
            )
        mark = "✅" if ans_l == test_ex.correct else "❌"
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
def cost(
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model for pricing lookup"),
    monthly_volume: int = typer.Option(100_000, help="Monthly API call volume"),
):
    """Show cost comparison across hosting tiers."""
    _cmd_cost(llm_model=llm_model, monthly_volume=monthly_volume)


def _cmd_cost(*, llm_model, monthly_volume):
    _banner()
    from pricing import quick_cost_table, LLM_PRICING

    if llm_model not in LLM_PRICING:
        console.print(f"[yellow]Unknown model '{llm_model}', using gpt-4o-mini pricing.[/yellow]")
        llm_model = "gpt-4o-mini"

    rows = quick_cost_table(llm_model, monthly_volume)

    table = Table(
        title=f"\nCost Comparison — {monthly_volume:,} calls/month with {llm_model}",
        box=box.ROUNDED, border_style="yellow",
    )
    table.add_column("SLM Hosting", style="bold")
    table.add_column("Pure LLM", justify="right")
    table.add_column("Hybrid", justify="right")
    table.add_column("Savings", justify="right", style="green")

    for r in rows:
        table.add_row(
            r["hosting"],
            f"${r['pure_llm_cost']:,.2f}",
            f"${r['hybrid_total']:,.2f}",
            f"${r['savings_abs']:,.2f} ({r['savings_pct']:.0f}%)",
        )
    console.print(table)
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
    total_examples = len(task.train_pool) + len(task.test_set)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        ptask = progress.add_task("Generating rationales...", total=total_examples)

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

    del model_orig, tok_orig
    import gc; gc.collect()

    console.print("[bold magenta]Phase 4[/bold magenta] — Multi-task dataset + LoRA fine-tuning")
    console.print(f"  Loss: L = {lambda_weight} * L_label + {1-lambda_weight:.2f} * L_rationale")

    train_annotated = annotated[:n_train]

    with console.status("[bold green]Tokenizing dataset..."):
        from slm import load_model as _lm
        _, tok_tmp, _ = _lm(slm)
        dataset = DistillDataset(
            train_annotated, task.instruction, task.choices,
            tok_tmp, max_len=512, lambda_weight=lambda_weight, seed=seed,
        )
    console.print(f"  Dataset: {len(dataset)} examples\n")

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
        f"[bold]Method:[/bold] Dataset mixing (equivalent to explicit loss weighting in expectation)\n"
        f"[bold]Adaptation:[/bold] LoRA (rank {8}, alpha {16}) — ~1-3% of parameters trainable\n"
        f"[bold]Teacher:[/bold] {llm_model} → rationales + labels\n"
        f"[bold]Student:[/bold] {_short(slm)} → trained to predict labels AND generate rationales",
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


if __name__ == "__main__":
    app()
