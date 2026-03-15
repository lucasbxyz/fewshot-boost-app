"""FewShot Boost — Interactive benchmark dashboard and live inference demo.

Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

_STRAT = {
    "zero_shot": "Zero-Shot",
    "random": "Random",
    "llm_assisted": "LLM-Assisted",
}
_STRAT_ORDER = ["Zero-Shot", "Random", "LLM-Assisted"]
_COLORS = {
    "Zero-Shot": "#94a3b8",
    "Random": "#f59e0b",
    "LLM-Assisted": "#10b981",
}

_SLM_OPTIONS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2b-it",
]


def _short(name: str) -> str:
    """HuggingFaceTB/SmolLM2-135M-Instruct -> SmolLM2-135M"""
    return name.split("/")[-1].replace("-Instruct", "")


# ── Data loading (always available) ──────────────────────────────────────────


@st.cache_data
def _load_summary() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "all_results_summary.csv")
    df["Strategy"] = df["strategy"].map(_STRAT)
    df["Model"] = df["slm_model"].apply(_short)
    return df


@st.cache_data
def _load_per_question() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "all_results_per_question.csv")


# ── Optional ML imports (for live inference only) ────────────────────────────

_ML = False
try:
    import torch  # noqa: F401

    _ML = True
except ImportError:
    pass

if _ML:

    @st.cache_resource
    def _load_slm(name: str):
        from slm import load_model

        return load_model(name)

    @st.cache_data
    def _load_boolq(n_train: int, n_test: int, seed: int):
        from benchmark import load_boolq

        return load_boolq(n_train, n_test, seed)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="FewShot Boost", page_icon="⚡", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# ⚡ FewShot Boost")
    st.caption("Intelligent few-shot selection for small language models")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔬 Live Demo", "🧪 Benchmark Lab", "🎓 Distillation"],
        label_visibility="collapsed",
    )
    st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════


def page_dashboard():
    st.title("Benchmark Results")
    st.markdown(
        "Comparing **zero-shot**, **random few-shot**, and **LLM-assisted few-shot** "
        "selection strategies on the "
        "[BoolQ](https://github.com/google-research-datasets/boolean-questions) "
        "yes/no question-answering benchmark."
    )

    df = _load_summary()
    pq = _load_per_question()

    # ── Sidebar filters ──
    with st.sidebar:
        st.subheader("Filters")
        all_models = sorted(df["Model"].unique())
        sel_models = st.multiselect("SLM Models", all_models, default=all_models)

        all_sel = sorted(df["llm_selector"].unique())
        sel_sel = st.multiselect("LLM Selectors", all_sel, default=all_sel)

        all_seeds = sorted(df["seed"].unique())
        sel_seeds = st.multiselect("Seeds", all_seeds, default=all_seeds)

    filt = df[
        df["Model"].isin(sel_models)
        & df["llm_selector"].isin(sel_sel)
        & df["seed"].isin(sel_seeds)
    ]

    if filt.empty:
        st.warning("No data matches current filters. Adjust in the sidebar.")
        return

    # ── Hero metrics ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    best_row = filt.loc[filt["mean_accuracy"].idxmax()]
    c1.metric(
        "Best Accuracy",
        f"{best_row['mean_accuracy']:.1%}",
        _short(best_row["slm_model"]),
    )

    boosts = []
    for _, r in filt[filt["strategy"] == "llm_assisted"].iterrows():
        zs = filt[
            (filt["slm_model"] == r["slm_model"])
            & (filt["seed"] == r["seed"])
            & (filt["strategy"] == "zero_shot")
        ]
        if not zs.empty:
            boosts.append(r["mean_accuracy"] - zs.iloc[0]["mean_accuracy"])
    max_boost = max(boosts) if boosts else 0
    c2.metric("Max LLM Boost", f"+{max_boost:.1%}", "over zero-shot")

    c3.metric("Configurations", str(len(filt)), "model × strategy × seed")
    c4.metric("Total Predictions", f"{len(pq):,}", "individual Q&A pairs")

    st.divider()

    # ── 1. Strategy Comparison ───────────────────────────────────────────────
    st.subheader("Strategy Comparison")

    agg = (
        filt.groupby(["Model", "Strategy"])
        .agg(acc=("mean_accuracy", "mean"), std=("std", "mean"))
        .reset_index()
    )

    fig1 = px.bar(
        agg,
        x="Model",
        y="acc",
        color="Strategy",
        barmode="group",
        error_y="std",
        color_discrete_map=_COLORS,
        category_orders={"Strategy": _STRAT_ORDER},
        labels={"acc": "Mean Accuracy", "Model": "Small Language Model"},
    )
    fig1.update_layout(
        template="plotly_white",
        height=420,
        yaxis_tickformat=".0%",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── 2. Two-column: LLM Selector + Model Size ────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("LLM Selector Effectiveness")
        llm_only = filt[filt["strategy"] == "llm_assisted"]
        if llm_only.empty:
            st.info("No LLM-assisted data in current filters.")
        else:
            agg2 = (
                llm_only.groupby(["llm_selector", "Model"])
                .agg(acc=("mean_accuracy", "mean"))
                .reset_index()
            )
            fig2 = px.bar(
                agg2,
                x="llm_selector",
                y="acc",
                color="Model",
                barmode="group",
                labels={"llm_selector": "LLM Selector", "acc": "Mean Accuracy"},
            )
            fig2.update_layout(
                template="plotly_white", height=380, yaxis_tickformat=".0%"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Accuracy by Model Size")
        size_order = [
            s
            for s in ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]
            if s in filt["Model"].values
        ]
        if len(size_order) < 2:
            st.info("Select multiple model sizes to see scaling trends.")
        else:
            agg3 = (
                filt.groupby(["Model", "Strategy"])
                .agg(acc=("mean_accuracy", "mean"))
                .reset_index()
            )
            agg3 = agg3[agg3["Model"].isin(size_order)]
            fig3 = px.line(
                agg3,
                x="Model",
                y="acc",
                color="Strategy",
                markers=True,
                color_discrete_map=_COLORS,
                category_orders={"Strategy": _STRAT_ORDER, "Model": size_order},
                labels={"acc": "Mean Accuracy", "Model": ""},
            )
            fig3.update_layout(
                template="plotly_white", height=380, yaxis_tickformat=".0%"
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── 3. Seed Stability Heatmap ────────────────────────────────────────────
    st.subheader("Reproducibility Across Seeds")
    heat = filt.copy()
    heat["Config"] = heat["Model"] + " · " + heat["Strategy"]
    pivot = heat.pivot_table(
        index="Config", columns="seed", values="mean_accuracy", aggfunc="mean"
    )
    pivot = pivot.sort_index()

    fig4 = px.imshow(
        pivot,
        text_auto=".1%",
        aspect="auto",
        labels=dict(x="Seed", y="", color="Accuracy"),
        color_continuous_scale="Greens",
    )
    fig4.update_layout(height=max(350, len(pivot) * 35 + 100))
    st.plotly_chart(fig4, use_container_width=True)

    # ── About ────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this benchmark"):
        st.markdown(
            "**FewShot Boost** evaluates whether using a large language model (LLM) "
            "to intelligently select few-shot examples can improve a small language "
            "model's (SLM) accuracy on yes/no question answering.\n\n"
            "**Three strategies:**\n"
            "- **Zero-Shot** — SLM answers with no examples\n"
            "- **Random** — SLM receives *k* randomly-selected examples\n"
            "- **LLM-Assisted** — An LLM picks the *k* most relevant examples "
            "for each question\n\n"
            "**Key finding:** LLM-assisted selection consistently outperforms "
            "random selection, with up to **+14.2 percentage points** improvement "
            "over zero-shot baselines."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE DEMO
# ═════════════════════════════════════════════════════════════════════════════


def _show_result(answer: str, raw: str, ground_truth: str | None, shots: list):
    """Render the result card for one strategy."""
    if ground_truth:
        correct = answer == ground_truth
        icon = "✅" if correct else "❌"
        st.markdown(f"## {icon} {answer}")
    else:
        st.markdown(f"## {answer}")

    if shots:
        with st.expander(f"Selected examples ({len(shots)})"):
            for i, s in enumerate(shots, 1):
                label = "yes" if s.answer else "no"
                st.markdown(f"**{i}.** _{s.question[:100]}_... → **{label}**")

    with st.expander("Raw model output"):
        st.code(raw if raw else "(empty)")


def page_live_demo():
    st.title("Live Inference Demo")

    if not _ML:
        st.warning("**Live inference is unavailable** — PyTorch is not installed.")
        st.markdown(
            "This feature requires local dependencies. Install them and relaunch:\n\n"
            "```bash\n"
            "pip install torch transformers datasets accelerate anthropic openai\n"
            "streamlit run app.py\n"
            "```\n\n"
            "The **📊 Dashboard** tab works everywhere (including Streamlit Cloud) "
            "without these dependencies."
        )
        return

    st.markdown(
        "Pick a BoolQ question (or write your own), then see how all three "
        "strategies affect the small model's answer — side by side."
    )

    # ── Sidebar config ───────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Inference Settings")
        slm = st.selectbox("SLM Model", _SLM_OPTIONS, format_func=_short)
        n_shots = st.slider("Few-shot examples (k)", 1, 5, 3)
        seed = st.number_input("Seed", value=42, min_value=0)
        st.divider()
        st.subheader("LLM Selector")
        provider = st.selectbox("Provider", ["openai", "anthropic"])
        default_llm = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"
        llm_model = st.text_input("Model", value=default_llm)
        api_key = st.text_input(
            "API Key", type="password", help="Required for LLM-Assisted strategy"
        )

    # ── Load resources ───────────────────────────────────────────────────────
    with st.status("Loading model & dataset...", expanded=False) as status:
        st.write(f"Loading **{_short(slm)}**...")
        model, tokenizer, device = _load_slm(slm)
        st.write("Loading BoolQ dataset...")
        train_pool, test_set = _load_boolq(50, 100, seed)
        status.update(
            label=f"Ready — {_short(slm)} on {device}  ·  "
            f"{len(train_pool)} train / {len(test_set)} test examples",
            state="complete",
        )

    # ── Question input ───────────────────────────────────────────────────────
    st.subheader("Question")

    if "passage" not in st.session_state:
        st.session_state.passage = ""
        st.session_state.question = ""
        st.session_state.ground_truth = None

    if st.button("🎲 Load Random BoolQ Question"):
        import random as _rng

        ex = _rng.choice(test_set)
        st.session_state.passage = ex.passage
        st.session_state.question = ex.question
        st.session_state.ground_truth = "yes" if ex.answer else "no"
        st.rerun()

    passage = st.text_area(
        "Passage",
        value=st.session_state.passage,
        height=120,
        placeholder="Paste a passage here...",
    )
    question = st.text_input(
        "Yes/No Question",
        value=st.session_state.question,
        placeholder="Does the passage support a yes or no answer?",
    )

    gt = st.session_state.ground_truth
    if gt:
        st.caption(f"Ground truth: **{gt}**")

    # ── Run ──────────────────────────────────────────────────────────────────
    if st.button(
        "🚀 Run All Three Strategies", type="primary", use_container_width=True
    ):
        if not passage.strip() or not question.strip():
            st.error("Enter both a passage and a question.")
            return

        from benchmark import BoolQExample
        from slm import generate_answer
        from selector import RandomSelector, create_llm_selector

        test_ex = BoolQExample(
            question_id="demo",
            question=question,
            passage=passage,
            answer=True,
            category="demo",
        )

        cols = st.columns(3)

        with cols[0]:
            st.markdown("### Zero-Shot")
            with st.spinner("Generating..."):
                ans, raw = generate_answer(model, tokenizer, device, test_ex, [])
            _show_result(ans, raw, gt, [])

        with cols[1]:
            st.markdown("### Random")
            with st.spinner("Selecting & generating..."):
                shots = RandomSelector().select(
                    train_pool, test_ex, n_shots, seed=seed
                )
                ans, raw = generate_answer(model, tokenizer, device, test_ex, shots)
            _show_result(ans, raw, gt, shots)

        with cols[2]:
            st.markdown("### LLM-Assisted")
            if not api_key:
                st.warning("Enter an API key in the sidebar to enable this strategy.")
            else:
                try:
                    with st.spinner("LLM selecting examples..."):
                        llm_sel = create_llm_selector(provider, llm_model, api_key)
                        shots = llm_sel.select(
                            train_pool, test_ex, n_shots, seed=seed
                        )
                    with st.spinner("Generating..."):
                        ans, raw = generate_answer(
                            model, tokenizer, device, test_ex, shots
                        )
                    _show_result(ans, raw, gt, shots)
                except Exception as e:
                    st.error(f"LLM selector failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  BENCHMARK LAB
# ═════════════════════════════════════════════════════════════════════════════


def page_benchmark_lab():
    st.title("Benchmark Lab")

    if not _ML:
        st.warning("**Benchmark Lab requires local dependencies** — PyTorch is not installed.")
        st.markdown(
            "Install inference dependencies and relaunch:\n\n"
            "```bash\n"
            "pip install torch transformers datasets accelerate anthropic openai\n"
            "streamlit run app.py\n"
            "```"
        )
        return

    st.markdown(
        "Pick any SLM, choose a benchmark (or define your own task), "
        "run the evaluation, and get a cost comparison."
    )

    from tasks import BENCHMARK_REGISTRY, load_task, build_custom_task
    from pricing import CostEstimate, LLM_PRICING, SLM_HOSTING

    # ── Sidebar config ───────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Model Selection")
        slm_preset = st.selectbox(
            "SLM Model", _SLM_OPTIONS + ["(custom)"], format_func=lambda x: _short(x) if x != "(custom)" else "Custom HuggingFace ID"
        )
        if slm_preset == "(custom)":
            slm = st.text_input("HuggingFace model ID", placeholder="org/model-name")
        else:
            slm = slm_preset

        st.divider()
        st.subheader("LLM Selector")
        provider = st.selectbox("Provider", ["openai", "anthropic"], key="lab_provider")
        default_llm = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"
        llm_model = st.text_input("LLM Model", value=default_llm, key="lab_llm")
        api_key = st.text_input("API Key", type="password", key="lab_key",
                                help="Required for LLM-Assisted strategy")
        st.divider()
        st.subheader("Run Config")
        n_shots = st.slider("Few-shot examples (k)", 1, 5, 3, key="lab_shots")
        sample_size = st.slider("Test questions", 5, 100, 20, key="lab_sample")
        n_train = st.slider("Training pool size", 10, 200, 50, key="lab_train")
        seed = st.number_input("Seed", value=42, min_value=0, key="lab_seed")

    # ── Benchmark selection ──────────────────────────────────────────────
    st.subheader("1. Choose a Benchmark")
    bench_mode = st.radio("", ["Pre-built benchmark", "Custom task template"],
                          horizontal=True, label_visibility="collapsed")

    task = None

    if bench_mode == "Pre-built benchmark":
        bench_key = st.selectbox(
            "Benchmark",
            list(BENCHMARK_REGISTRY.keys()),
            format_func=lambda k: BENCHMARK_REGISTRY[k]["label"],
        )
        info = BENCHMARK_REGISTRY[bench_key]
        st.caption(f"{info['description']}  —  Classes: **{', '.join(info['choices'])}**")

        if st.button("📥 Load benchmark dataset", key="load_bench"):
            with st.spinner(f"Loading {info['label']}..."):
                task = load_task(bench_key, n_train=n_train, n_test=sample_size, seed=seed)
            st.session_state["lab_task"] = task

        if "lab_task" in st.session_state:
            task = st.session_state["lab_task"]
            st.success(f"**{task.name}** loaded — {len(task.train_pool)} train, {len(task.test_set)} test examples")
    else:
        with st.form("custom_task_form"):
            task_name = st.text_input("Task name", placeholder="Spam Detection")
            task_instr = st.text_area("Instruction", placeholder="Classify whether this email is spam or legitimate.")
            task_choices_str = st.text_input("Answer choices (comma-separated)", placeholder="spam, legitimate")
            task_examples = st.text_area(
                "Examples (one per line: `input text ||| label`)",
                height=200,
                placeholder="Great product, highly recommend! ||| positive\nTerrible waste of money ||| negative",
            )
            submitted = st.form_submit_button("Build task")

        if submitted and task_name and task_instr and task_choices_str and task_examples:
            choices = [c.strip() for c in task_choices_str.split(",") if c.strip()]
            try:
                task = build_custom_task(task_name, task_instr, choices, task_examples, seed=seed)
                st.session_state["lab_task"] = task
                st.success(f"**{task.name}** built — {len(task.train_pool)} train, {len(task.test_set)} test")
            except ValueError as e:
                st.error(str(e))

        if "lab_task" in st.session_state and task is None:
            task = st.session_state.get("lab_task")

    if task is None:
        st.info("Load a benchmark or build a custom task above to continue.")
        return

    # ── Run benchmark ────────────────────────────────────────────────────
    st.subheader("2. Run Evaluation")

    if not slm:
        st.warning("Select or enter an SLM model in the sidebar.")
        return

    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True, key="run_bench"):
        from slm import load_model, generate_answer_generic
        from selector import RandomSelector, GenericLLMSelector

        progress = st.progress(0, text="Loading model...")
        model, tokenizer, device = _load_slm(slm)

        test_set = task.test_set
        n = len(test_set)
        strategies = ["Zero-Shot", "Random"]
        if api_key:
            strategies.append("LLM-Assisted")

        results = {s: [] for s in strategies}
        total_slm_tokens = 0

        llm_sel = None
        if "LLM-Assisted" in strategies:
            llm_sel = GenericLLMSelector(
                provider, llm_model, api_key,
                task_instruction=task.instruction, task_choices=task.choices,
            )

        random_sel = RandomSelector()

        for i, test_ex in enumerate(test_set):
            pct = (i + 1) / n
            progress.progress(pct, text=f"Question {i+1}/{n}...")

            # Zero-shot
            ans, _, tok = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, [],
            )
            results["Zero-Shot"].append(ans == test_ex.correct)
            total_slm_tokens += tok

            # Random
            shots = random_sel.select(task.train_pool, test_ex, n_shots, seed=seed + i)
            ans, _, tok = generate_answer_generic(
                model, tokenizer, device,
                task.instruction, task.choices,
                test_ex.input_text, test_ex.context, shots,
            )
            results["Random"].append(ans == test_ex.correct)
            total_slm_tokens += tok

            # LLM-Assisted
            if llm_sel:
                shots = llm_sel.select(task.train_pool, test_ex, n_shots, seed=seed + i)
                ans, _, tok = generate_answer_generic(
                    model, tokenizer, device,
                    task.instruction, task.choices,
                    test_ex.input_text, test_ex.context, shots,
                )
                results["LLM-Assisted"].append(ans == test_ex.correct)
                total_slm_tokens += tok

        progress.progress(1.0, text="Done!")

        accuracies = {s: sum(r) / len(r) if r else 0.0 for s, r in results.items()}

        cost_est = CostEstimate(
            llm_input_tokens=llm_sel.total_input_tokens if llm_sel else 0,
            llm_output_tokens=llm_sel.total_output_tokens if llm_sel else 0,
            slm_input_tokens=total_slm_tokens,
            n_questions=n,
            accuracy_zero_shot=accuracies.get("Zero-Shot", 0),
            accuracy_random=accuracies.get("Random", 0),
            accuracy_llm_assisted=accuracies.get("LLM-Assisted", 0),
        )

        st.session_state["lab_results"] = accuracies
        st.session_state["lab_cost_est"] = cost_est
        st.session_state["lab_llm_model"] = llm_model
        st.session_state["lab_n"] = n

    # ── Results panel ────────────────────────────────────────────────────
    if "lab_results" in st.session_state:
        accuracies = st.session_state["lab_results"]
        cost_est = st.session_state["lab_cost_est"]
        llm_used = st.session_state["lab_llm_model"]
        n = st.session_state["lab_n"]

        st.subheader("3. Results")
        cols = st.columns(len(accuracies))
        for col, (strat, acc) in zip(cols, accuracies.items()):
            color = _COLORS.get(strat, "#6b7280")
            col.metric(strat, f"{acc:.1%}")

        import plotly.graph_objects as go
        fig = go.Figure()
        for strat, acc in accuracies.items():
            fig.add_trace(go.Bar(
                name=strat, x=[strat], y=[acc],
                marker_color=_COLORS.get(strat, "#6b7280"),
                text=f"{acc:.1%}", textposition="outside",
            ))
        fig.update_layout(
            template="plotly_white", height=350, showlegend=False,
            yaxis_tickformat=".0%", yaxis_title="Accuracy",
            title=f"Strategy Comparison on {task.name} ({n} questions)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Cost estimation ──────────────────────────────────────────────
        st.subheader("4. Cost Estimation")

        st.caption(
            f"Measured: {cost_est.llm_input_tokens:,} LLM input tokens, "
            f"{cost_est.llm_output_tokens:,} LLM output tokens across {n} questions."
        )

        vol = st.slider("Monthly API call volume", 1_000, 1_000_000, 100_000,
                         step=1_000, key="lab_vol", format="%d")

        hosting_key = st.selectbox(
            "SLM hosting option",
            list(SLM_HOSTING.keys()),
            format_func=lambda k: SLM_HOSTING[k]["label"],
            key="lab_hosting",
        )

        proj = cost_est.project_monthly(vol, llm_used, hosting_key)

        m1, m2, m3 = st.columns(3)
        m1.metric("Pure LLM Cost", f"${proj['pure_llm_cost']:,.2f}/mo")
        m2.metric("Hybrid Cost", f"${proj['hybrid_total']:,.2f}/mo")
        m3.metric(
            "Monthly Savings",
            f"${proj['savings_abs']:,.2f}",
            f"{proj['savings_pct']:.0f}%",
        )

        with st.expander("Cost breakdown"):
            st.markdown(
                f"| Component | Cost |\n"
                f"|---|---|\n"
                f"| Pure LLM ({llm_used}) | ${proj['pure_llm_cost']:,.2f}/mo |\n"
                f"| Hybrid: LLM selector calls | ${proj['hybrid_llm_cost']:,.2f}/mo |\n"
                f"| Hybrid: SLM hosting | ${proj['hybrid_hosting_cost']:,.0f}/mo |\n"
                f"| **Hybrid total** | **${proj['hybrid_total']:,.2f}/mo** |\n"
                f"| **Savings** | **${proj['savings_abs']:,.2f}/mo ({proj['savings_pct']:.0f}%)** |"
            )
            if proj["break_even_volume"] > 0:
                st.caption(f"Break-even at ~{proj['break_even_volume']:,} calls/month (to cover hosting costs).")

        with st.expander("Accuracy trade-off"):
            st.markdown(
                f"| Approach | Accuracy |\n"
                f"|---|---|\n"
                f"| Pure LLM (assumed) | ~90% |\n"
                f"| Hybrid: Zero-Shot SLM | {cost_est.accuracy_zero_shot:.1%} |\n"
                f"| Hybrid: Random few-shot | {cost_est.accuracy_random:.1%} |\n"
                f"| Hybrid: LLM-Assisted few-shot | {cost_est.accuracy_llm_assisted:.1%} |"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  DISTILLATION LAB
# ═════════════════════════════════════════════════════════════════════════════


def page_distillation():
    st.title("Distillation Lab")
    st.markdown(
        "**Distilling Step-by-Step** (Hsieh et al., 2023): the teacher LLM generates "
        "labels *and* rationales, then the student SLM is LoRA-fine-tuned with a "
        "multi-task loss:  `L = λ · L_label + (1-λ) · L_rationale`"
    )

    if not _ML:
        st.warning("**Requires local dependencies** — PyTorch is not installed.")
        st.markdown(
            "```bash\n"
            "pip install torch transformers datasets accelerate anthropic openai peft\n"
            "```"
        )
        return

    from tasks import BENCHMARK_REGISTRY, load_task

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Distillation Config")
        bench_key = st.selectbox(
            "Benchmark", list(BENCHMARK_REGISTRY.keys()),
            format_func=lambda k: BENCHMARK_REGISTRY[k]["label"], key="dist_bench",
        )
        slm = st.selectbox("Student SLM", _SLM_OPTIONS[:3], format_func=_short, key="dist_slm")
        st.divider()
        st.subheader("Teacher LLM")
        provider = st.selectbox("Provider", ["openai", "anthropic"], key="dist_prov")
        default_llm = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"
        llm_model = st.text_input("Model", value=default_llm, key="dist_llm")
        api_key = st.text_input("API Key", type="password", key="dist_key")
        st.divider()
        st.subheader("Training")
        lambda_w = st.slider("λ (label weight)", 0.0, 1.0, 0.5, 0.05,
                             key="dist_lambda",
                             help="λ=1.0 → only labels, λ=0.0 → only rationales")
        epochs = st.slider("Epochs", 1, 10, 3, key="dist_epochs")
        lr = st.select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4, key="dist_lr")
        n_train = st.slider("Training examples", 10, 100, 30, key="dist_ntrain")
        n_test = st.slider("Test examples", 5, 50, 20, key="dist_ntest")
        seed = st.number_input("Seed", value=42, key="dist_seed")

    info = BENCHMARK_REGISTRY[bench_key]
    st.caption(f"**{info['label']}** — {info['description']}  ·  Classes: {', '.join(info['choices'])}")

    # ── Phase 1: Generate rationales ─────────────────────────────────────
    st.subheader("Phase 1 — Teacher Rationale Generation")
    st.markdown(
        "The teacher LLM produces a **label + natural language rationale** for each training example."
    )

    if st.button("🧠 Generate Rationales", key="gen_rat"):
        if not api_key:
            st.error("Enter an API key in the sidebar.")
            return

        from distill import generate_teacher_rationales

        with st.spinner("Loading benchmark..."):
            task = load_task(bench_key, n_train=n_train, n_test=n_test, seed=seed)
        st.session_state["dist_task"] = task

        progress = st.progress(0, text="Generating rationales...")

        def _cb(i, total):
            progress.progress(i / total, text=f"Rationale {i}/{total}...")

        annotated = generate_teacher_rationales(task, provider, llm_model, api_key, callback=_cb)
        progress.progress(1.0, text="Done!")
        st.session_state["dist_annotated"] = annotated
        st.session_state["dist_n_train"] = n_train

    if "dist_annotated" in st.session_state:
        annotated = st.session_state["dist_annotated"]
        st.success(f"{len(annotated)} examples annotated with rationales.")

        with st.expander("Sample rationales (first 5)"):
            for ex in annotated[:5]:
                st.markdown(
                    f"**Q:** {ex.input_text[:120]}...\n\n"
                    f"**Label:** {ex.label}  ·  **Rationale:** _{ex.rationale}_\n\n---"
                )
    else:
        st.info("Click above to generate rationales from the teacher LLM.")
        return

    # ── Phase 2: Evaluate baseline ───────────────────────────────────────
    st.subheader("Phase 2 — Baseline (Original Student)")

    if st.button("📏 Evaluate baseline (no distillation)", key="eval_base"):
        from distill import evaluate_distilled

        task = st.session_state["dist_task"]
        with st.status("Loading original model...", expanded=False):
            model, tokenizer, device = _load_slm(slm)

        progress = st.progress(0, text="Evaluating baseline...")
        def _cb(i, t): progress.progress(i / t, text=f"Question {i}/{t}...")

        baseline = evaluate_distilled(model, tokenizer, device, task, callback=_cb)
        progress.progress(1.0, text="Done!")
        st.session_state["dist_baseline"] = baseline

    if "dist_baseline" in st.session_state:
        b = st.session_state["dist_baseline"]
        st.metric("Baseline Accuracy (zero-shot)", f"{b['accuracy']:.1%}",
                  f"{b['n_correct']}/{b['n_total']} correct")
    else:
        st.info("Click above to measure the baseline before distillation.")
        return

    # ── Phase 3: Fine-tune ───────────────────────────────────────────────
    st.subheader("Phase 3 — LoRA Fine-Tuning")
    st.markdown(
        f"Multi-task loss with **λ = {lambda_w}**: "
        f"{lambda_w:.0%} label prediction, {1-lambda_w:.0%} rationale generation."
    )

    if st.button("🔥 Distill Student", type="primary", key="distill_btn"):
        from distill import DistillDataset, fine_tune_student

        task = st.session_state["dist_task"]
        annotated = st.session_state["dist_annotated"]

        n_tr = st.session_state.get("dist_n_train", n_train)
        train_annotated = annotated[:n_tr]

        with st.status("Preparing dataset...", expanded=True) as status:
            st.write("Tokenizing multi-task training data...")
            from slm import load_model as _lm
            _, tok_tmp, _ = _load_slm(slm)

            ds = DistillDataset(
                train_annotated, task.instruction, task.choices,
                tok_tmp, max_len=512, lambda_weight=lambda_w, seed=seed,
            )
            st.write(f"Dataset: {len(ds)} examples")
            status.update(label=f"Dataset ready ({len(ds)} examples)", state="complete")

        progress = st.progress(0, text="Fine-tuning...")
        loss_history = []

        def _cb(step, total, loss_val):
            progress.progress(step / total, text=f"Step {step}/{total}  loss={loss_val:.4f}")
            loss_history.append(loss_val)

        model_d, tok_d, dev_d = fine_tune_student(
            slm, ds, epochs=epochs, lr=lr, callback=_cb,
        )
        progress.progress(1.0, text="Fine-tuning complete!")

        st.session_state["dist_model"] = (model_d, tok_d, dev_d)
        st.session_state["dist_loss_history"] = loss_history

    if "dist_loss_history" in st.session_state:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state["dist_loss_history"], mode="lines",
            name="Training Loss", line=dict(color="#10b981"),
        ))
        fig.update_layout(
            template="plotly_white", height=250,
            title="Training Loss", xaxis_title="Step", yaxis_title="Loss",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "dist_model" not in st.session_state:
        return

    # ── Phase 4: Evaluate distilled ──────────────────────────────────────
    st.subheader("Phase 4 — Evaluate Distilled Student")

    if st.button("📊 Evaluate distilled model", key="eval_dist"):
        from distill import evaluate_distilled

        task = st.session_state["dist_task"]
        model_d, tok_d, dev_d = st.session_state["dist_model"]

        progress = st.progress(0, text="Evaluating distilled model...")
        def _cb(i, t): progress.progress(i / t, text=f"Question {i}/{t}...")

        distilled = evaluate_distilled(model_d, tok_d, dev_d, task, callback=_cb)
        progress.progress(1.0, text="Done!")
        st.session_state["dist_result"] = distilled

    if "dist_result" in st.session_state:
        d = st.session_state["dist_result"]
        b = st.session_state["dist_baseline"]

        delta = d["accuracy"] - b["accuracy"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline (Original)", f"{b['accuracy']:.1%}")
        c2.metric("Distilled", f"{d['accuracy']:.1%}",
                  f"{delta:+.1%}", delta_color="normal")
        c3.metric("Improvement", f"{delta:+.1%}",
                  f"{d['n_correct']}/{d['n_total']} correct")

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Original (zero-shot)", "Distilled (LoRA)"],
            y=[b["accuracy"], d["accuracy"]],
            marker_color=["#94a3b8", "#10b981"],
            text=[f"{b['accuracy']:.1%}", f"{d['accuracy']:.1%}"],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_white", height=350,
            yaxis_tickformat=".0%", yaxis_title="Accuracy",
            title=f"Distillation Result (λ={lambda_w})",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Loss function details"):
            st.markdown(
                f"**Multi-task loss:** `L = λ · L_label + (1-λ) · L_rationale`\n\n"
                f"- **λ = {lambda_w}** → {lambda_w:.0%} of training examples are label-only, "
                f"{1-lambda_w:.0%} are rationale+label\n"
                f"- Implemented via **dataset mixing** (equivalent to explicit loss weighting "
                f"in expectation)\n"
                f"- **LoRA** adapters fine-tuned (~1-3% of parameters)\n"
                f"- Standard **cross-entropy** on response tokens (prompt tokens masked with -100)"
            )


# ── Route ────────────────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    page_dashboard()
elif page == "🔬 Live Demo":
    page_live_demo()
elif page == "🧪 Benchmark Lab":
    page_benchmark_lab()
else:
    page_distillation()
