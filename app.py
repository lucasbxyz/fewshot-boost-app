"""David vs Goliath — Interactive benchmark dashboard and live inference demo.

Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CACHE_DIR = APP_DIR / "cache"

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


@st.cache_data
def _load_per_category() -> pd.DataFrame:
    path = DATA_DIR / "all_results_per_category.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["Strategy"] = df["strategy"].map(_STRAT)
        df["Model"] = df["slm_model"].apply(_short)
        return df
    return pd.DataFrame()


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


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="David vs Goliath", page_icon="DvG", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# David vs Goliath")
    st.caption("Cut your LLM API bill by up to 90%")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Dashboard", "Live Demo", "Benchmark Lab", "Distillation", "Your Savings"],
        label_visibility="collapsed",
    )
    st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════


def page_dashboard():
    st.title("Benchmark Results")
    st.markdown(
        "**20,000+ predictions** across multiple model sizes, LLM selectors, and seeds "
        "prove that intelligent example selection consistently lifts small-model accuracy — "
        "at a fraction of full-LLM cost."
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

    c3.metric("Configurations", str(len(filt)), "model x strategy x seed")
    c4.metric("Total Predictions", f"{len(pq):,}", "individual Q&A pairs")

    st.divider()

    # ── 1. Strategy Comparison (with CIs) ────────────────────────────────────
    st.subheader("Strategy Comparison")

    has_ci = "ci_lower" in filt.columns and "ci_upper" in filt.columns

    agg = (
        filt.groupby(["Model", "Strategy"])
        .agg(
            acc=("mean_accuracy", "mean"),
            ci_lo=("ci_lower", "mean") if has_ci else ("mean_accuracy", "mean"),
            ci_hi=("ci_upper", "mean") if has_ci else ("mean_accuracy", "mean"),
        )
        .reset_index()
    )
    if has_ci:
        agg["ci_err_minus"] = agg["acc"] - agg["ci_lo"]
        agg["ci_err_plus"] = agg["ci_hi"] - agg["acc"]

    import plotly.graph_objects as go
    fig1 = go.Figure()
    for strat in _STRAT_ORDER:
        sub = agg[agg["Strategy"] == strat]
        kwargs = {}
        if has_ci:
            kwargs["error_y"] = dict(
                type="data",
                symmetric=False,
                array=sub["ci_err_plus"].tolist(),
                arrayminus=sub["ci_err_minus"].tolist(),
            )
        fig1.add_trace(go.Bar(
            x=sub["Model"], y=sub["acc"], name=strat,
            marker_color=_COLORS.get(strat, "#94a3b8"),
            **kwargs,
        ))
    fig1.update_layout(
        template="plotly_white", height=420, barmode="group",
        yaxis_tickformat=".0%", yaxis_title="Mean Accuracy (95% CI)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
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
        all_short = sorted(filt["Model"].unique())
        if len(all_short) < 2:
            st.info("Select multiple model sizes to see scaling trends.")
        else:
            agg3 = (
                filt.groupby(["Model", "Strategy"])
                .agg(acc=("mean_accuracy", "mean"))
                .reset_index()
            )
            fig3 = px.line(
                agg3,
                x="Model",
                y="acc",
                color="Strategy",
                markers=True,
                color_discrete_map=_COLORS,
                category_orders={"Strategy": _STRAT_ORDER, "Model": all_short},
                labels={"acc": "Mean Accuracy", "Model": ""},
            )
            fig3.update_layout(
                template="plotly_white", height=380, yaxis_tickformat=".0%"
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── 3. Seed Stability Heatmap ────────────────────────────────────────────
    st.subheader("Reproducibility Across Seeds")
    heat = filt.copy()
    heat["Config"] = heat["Model"] + " / " + heat["Strategy"]
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

    # ── 4. Per-Category Breakdown ────────────────────────────────────────────
    cat_df = _load_per_category()
    if not cat_df.empty:
        st.subheader("Per-Category Breakdown")
        cat_filt = cat_df[
            cat_df["Model"].isin(sel_models)
            & cat_df["llm_selector"].isin(sel_sel)
            & cat_df["seed"].isin(sel_seeds)
        ]
        if not cat_filt.empty and "category" in cat_filt.columns:
            cat_agg = (
                cat_filt.groupby(["category", "Strategy"])
                .agg(acc=("mean_accuracy", "mean"), n=("n_questions", "sum"))
                .reset_index()
                .sort_values("acc", ascending=False)
            )
            top_cats = cat_agg.groupby("category")["n"].sum().nlargest(12).index
            cat_agg = cat_agg[cat_agg["category"].isin(top_cats)]

            fig5 = px.bar(
                cat_agg, x="category", y="acc", color="Strategy",
                barmode="group", color_discrete_map=_COLORS,
                category_orders={"Strategy": _STRAT_ORDER},
                labels={"acc": "Accuracy", "category": "Category"},
            )
            fig5.update_layout(
                template="plotly_white", height=400, yaxis_tickformat=".0%",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig5, use_container_width=True)

    # ── 5. Per-Question Analysis ──────────────────────────────────────────────
    with st.expander("Per-Question Analysis"):
        pq_filt = pq[
            pq["slm_model"].apply(_short).isin(sel_models)
            & pq["llm_selector"].isin(sel_sel)
            & pq["seed"].isin(sel_seeds)
        ]
        if not pq_filt.empty:
            q_acc = pq_filt.groupby("question_id")["correct"].mean().reset_index()
            q_acc.columns = ["question_id", "accuracy"]

            n_hard = (q_acc["accuracy"] < 0.3).sum()
            n_easy = (q_acc["accuracy"] > 0.8).sum()
            n_total = len(q_acc)

            h1, h2, h3 = st.columns(3)
            h1.metric("Total Questions", f"{n_total:,}")
            h2.metric("Hard Questions (<30% acc)", f"{n_hard}")
            h3.metric("Easy Questions (>80% acc)", f"{n_easy}")

            fig6 = px.histogram(
                q_acc, x="accuracy", nbins=20,
                labels={"accuracy": "Per-Question Accuracy", "count": "# Questions"},
                color_discrete_sequence=["#10b981"],
            )
            fig6.update_layout(
                template="plotly_white", height=300,
                xaxis_tickformat=".0%",
            )
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No per-question data matches current filters.")

    # ── About ────────────────────────────────────────────────────────────────
    with st.expander("About this benchmark"):
        st.markdown(
            "**David vs Goliath** evaluates whether using a large language model (LLM) "
            "to intelligently select few-shot examples can improve a small language "
            "model's (SLM) accuracy on classification tasks.\n\n"
            "**Three strategies:**\n"
            "- **Zero-Shot** -- SLM answers with no examples\n"
            "- **Random** -- SLM receives *k* randomly-selected examples\n"
            "- **LLM-Assisted** -- An LLM picks the *k* most relevant examples "
            "for each question\n\n"
            f"**Key finding:** LLM-assisted selection consistently outperforms "
            f"random selection, with up to **+{max_boost:.1%}** improvement "
            f"over zero-shot baselines."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE DEMO
# ═════════════════════════════════════════════════════════════════════════════


def _show_result(answer: str, raw: str, ground_truth: str | None, shots: list):
    """Render the result card for one strategy."""
    if ground_truth:
        correct = answer == ground_truth
        icon = "+" if correct else "x"
        st.markdown(f"## {icon} {answer}")
    else:
        st.markdown(f"## {answer}")

    if shots:
        with st.expander(f"Selected examples ({len(shots)})"):
            for i, s in enumerate(shots, 1):
                text = getattr(s, "input_text", getattr(s, "question", str(s)))
                label = getattr(s, "correct", "yes" if getattr(s, "answer", True) else "no")
                st.markdown(f"**{i}.** _{text[:100]}_... -> **{label}**")

    with st.expander("Raw model output"):
        st.code(raw if raw else "(empty)")


def page_live_demo():
    st.title("Live Inference Demo")

    if not _ML:
        st.warning("**Live inference is unavailable** -- PyTorch is not installed.")
        st.markdown(
            "This feature requires local dependencies. Install them and relaunch:\n\n"
            "```bash\n"
            "pip install torch transformers datasets accelerate anthropic openai\n"
            "streamlit run app.py\n"
            "```\n\n"
            "The **Dashboard** tab works everywhere (including Streamlit Cloud) "
            "without these dependencies."
        )
        return

    st.markdown(
        "Pick a BoolQ question (or write your own), then see how all three "
        "strategies affect the small model's answer -- side by side."
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

    from tasks import load_boolq_task, TaskExample
    from slm import generate_answer_generic
    from selector import RandomSelector, GenericLLMSelector

    # ── Load resources ───────────────────────────────────────────────────────
    with st.status("Loading model & dataset...", expanded=False) as status:
        st.write(f"Loading **{_short(slm)}**...")
        model, tokenizer, device = _load_slm(slm)
        st.write("Loading BoolQ dataset...")
        task = load_boolq_task(n_train=50, n_test=100, seed=seed)
        status.update(
            label=f"Ready -- {_short(slm)} on {device}  |  "
            f"{len(task.train_pool)} train / {len(task.test_set)} test examples",
            state="complete",
        )

    # ── Question input ───────────────────────────────────────────────────────
    st.subheader("Question")

    if "passage" not in st.session_state:
        st.session_state.passage = ""
        st.session_state.question = ""
        st.session_state.ground_truth = None

    if st.button("Load Random BoolQ Question"):
        import random as _rng

        ex = _rng.choice(task.test_set)
        st.session_state.passage = ex.context
        st.session_state.question = ex.input_text
        st.session_state.ground_truth = ex.correct
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
        "Run All Three Strategies", type="primary", use_container_width=True
    ):
        if not passage.strip() or not question.strip():
            st.error("Enter both a passage and a question.")
            return

        test_ex = TaskExample(
            id="demo",
            input_text=question,
            context=passage,
            correct=gt or "",
            category="demo",
        )

        cols = st.columns(3)

        with cols[0]:
            st.markdown("### Zero-Shot")
            with st.spinner("Generating..."):
                ans, raw, _ = generate_answer_generic(
                    model, tokenizer, device,
                    task.instruction, task.choices,
                    test_ex.input_text, test_ex.context, [],
                )
            _show_result(ans, raw, gt, [])

        with cols[1]:
            st.markdown("### Random")
            with st.spinner("Selecting & generating..."):
                shots = RandomSelector().select(
                    task.train_pool, test_ex, n_shots, seed=seed
                )
                ans, raw, _ = generate_answer_generic(
                    model, tokenizer, device,
                    task.instruction, task.choices,
                    test_ex.input_text, test_ex.context, shots,
                )
            _show_result(ans, raw, gt, shots)

        with cols[2]:
            st.markdown("### LLM-Assisted")
            if not api_key:
                st.warning("Enter an API key in the sidebar to enable this strategy.")
            else:
                try:
                    with st.spinner("LLM selecting examples..."):
                        llm_sel = GenericLLMSelector(
                            provider, llm_model, api_key,
                            task_instruction=task.instruction,
                            task_choices=task.choices,
                            cache_dir=CACHE_DIR / "selector",
                        )
                        shots = llm_sel.select(
                            task.train_pool, test_ex, n_shots, seed=seed
                        )
                    with st.spinner("Generating..."):
                        ans, raw, _ = generate_answer_generic(
                            model, tokenizer, device,
                            task.instruction, task.choices,
                            test_ex.input_text, test_ex.context, shots,
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
        st.warning("**Benchmark Lab requires local dependencies** -- PyTorch is not installed.")
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
    from pricing import CostEstimate, LLM_PRICING, SLM_HOSTING, project_monthly

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
        st.caption(f"{info['description']}  --  Classes: **{', '.join(info['choices'])}**")

        if st.button("Load benchmark dataset", key="load_bench"):
            with st.spinner(f"Loading {info['label']}..."):
                task = load_task(bench_key, n_train=n_train, n_test=sample_size, seed=seed)
            st.session_state["lab_task"] = task

        if "lab_task" in st.session_state:
            task = st.session_state["lab_task"]
            st.success(f"**{task.name}** loaded -- {len(task.train_pool)} train, {len(task.test_set)} test examples")
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
                st.success(f"**{task.name}** built -- {len(task.train_pool)} train, {len(task.test_set)} test")
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

    if st.button("Run Benchmark", type="primary", use_container_width=True, key="run_bench"):
        from slm import generate_answer_generic
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
                cache_dir=CACHE_DIR / "selector",
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
        st.subheader("4. Cost Estimation (token-based)")

        st.caption(
            f"**Measured from your run:** {cost_est.avg_llm_input_per_q:.0f} LLM input tokens/question, "
            f"{cost_est.avg_llm_output_per_q:.0f} LLM output tokens/question "
            f"({cost_est.llm_input_tokens:,} total across {n} questions)."
        )

        st.markdown("##### Your baseline (what you'd spend without David vs Goliath)")
        b1, b2 = st.columns(2)
        direct_model = b1.selectbox(
            "Your current LLM",
            list(LLM_PRICING.keys()),
            index=list(LLM_PRICING.keys()).index(llm_used) if llm_used in LLM_PRICING else 0,
            key="lab_direct_model",
        )
        direct_accuracy = b2.slider(
            "Your LLM's accuracy on this task", 0.5, 1.0, 0.90, 0.01,
            key="lab_direct_acc",
        )
        d1, d2 = st.columns(2)
        direct_input_tpq = d1.number_input(
            "Avg input tokens/question (your LLM)",
            min_value=10, max_value=10_000, value=350, step=50,
            key="lab_direct_in",
        )
        direct_output_tpq = d2.number_input(
            "Avg output tokens/question (your LLM)",
            min_value=1, max_value=2_000, value=20, step=5,
            key="lab_direct_out",
        )

        st.markdown("##### Scale & hosting")
        s1, s2 = st.columns(2)
        vol = s1.slider("Monthly question volume", 1_000, 1_000_000, 100_000,
                         step=1_000, key="lab_vol", format="%d")

        hosting_keys = [k for k in SLM_HOSTING.keys() if k != "custom"]
        hosting_keys.append("custom")
        hosting_key = s2.selectbox(
            "SLM hosting option",
            hosting_keys,
            format_func=lambda k: SLM_HOSTING[k]["label"],
            key="lab_hosting",
        )
        custom_hosting = 0.0
        if hosting_key == "custom":
            custom_hosting = st.number_input(
                "Custom monthly hosting cost ($)",
                min_value=0.0, max_value=10_000.0, value=50.0, step=10.0,
                key="lab_custom_hosting",
            )

        proj = project_monthly(
            vol,
            direct_model=direct_model,
            direct_input_tpq=float(direct_input_tpq),
            direct_output_tpq=float(direct_output_tpq),
            direct_accuracy=direct_accuracy,
            selector_model=llm_used,
            slm_hosting_key=hosting_key,
            custom_hosting_cost=custom_hosting,
            measured=cost_est,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Pure LLM Cost", f"${proj.pure_llm_cost:,.2f}/mo")
        m2.metric("Hybrid Cost", f"${proj.hybrid_total:,.2f}/mo")
        m3.metric(
            "Monthly Savings",
            f"${proj.savings_abs:,.2f}",
            f"{proj.savings_pct:.0f}%",
        )

        with st.expander("Full cost breakdown (all token-based)"):
            st.markdown(
                f"| Component | Per Question | Monthly ({vol:,} calls) |\n"
                f"|---|---|---|\n"
                f"| **Your LLM** ({direct_model}) | "
                f"{direct_input_tpq:.0f} in + {direct_output_tpq:.0f} out tokens | "
                f"**${proj.pure_llm_cost:,.2f}** |\n"
                f"| Hybrid: LLM selector ({llm_used}) | "
                f"{proj.selector_input_tpq:.0f} in + {proj.selector_output_tpq:.0f} out tokens | "
                f"${proj.hybrid_llm_cost:,.2f} |\n"
                f"| Hybrid: SLM hosting | -- | ${proj.hybrid_hosting_cost:,.0f} |\n"
                f"| **Hybrid total** | | **${proj.hybrid_total:,.2f}** |\n"
                f"| **Savings** | | **${proj.savings_abs:,.2f} ({proj.savings_pct:.0f}%)** |"
            )
            if proj.break_even_volume > 0:
                st.info(f"Break-even at **{proj.break_even_volume:,}** calls/month to cover hosting costs.")

        with st.expander("Accuracy trade-off"):
            st.markdown(
                f"| Approach | Accuracy |\n"
                f"|---|---|\n"
                f"| Your LLM ({direct_model}) | {direct_accuracy:.0%} |\n"
                f"| Hybrid: Zero-Shot SLM | {cost_est.accuracy_zero_shot:.1%} |\n"
                f"| Hybrid: Random few-shot | {cost_est.accuracy_random:.1%} |\n"
                f"| Hybrid: LLM-Assisted | {cost_est.accuracy_llm_assisted:.1%} |"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  DISTILLATION LAB
# ═════════════════════════════════════════════════════════════════════════════


def page_distillation():
    st.title("Distillation Lab")
    st.markdown(
        "**Distilling Step-by-Step** (Hsieh et al., 2023): the teacher LLM generates "
        "labels *and* rationales, then the student SLM is LoRA-fine-tuned with a "
        "multi-task loss:  `L = lambda * L_label + (1-lambda) * L_rationale`"
    )

    if not _ML:
        st.info(
            "**Live distillation requires local hardware** (PyTorch + GPU/MPS). "
            "Install with: `pip install torch transformers datasets accelerate anthropic openai peft`"
        )
        st.markdown("---")
        st.subheader("How Distillation Works")
        st.markdown(
            "1. **Teacher generates rationales** -- The LLM produces a label *and* a natural language "
            "explanation for each training example\n"
            "2. **Multi-task dataset** -- Examples are split into label-only and rationale+label tasks "
            "at ratio lambda : (1-lambda)\n"
            "3. **LoRA fine-tuning** -- The student SLM is fine-tuned with ~1-3% trainable parameters\n"
            "4. **Evaluation** -- The distilled student is compared against its original zero-shot performance\n\n"
            "**To run distillation online**, deploy this app to a "
            "[HuggingFace Space](https://huggingface.co/spaces) with GPU (T4, free tier available). "
            "A Dockerfile is included in this repo for one-click deployment."
        )

        # Show pre-computed distillation results if available
        distill_results_path = DATA_DIR / "distillation_results.csv"
        if distill_results_path.exists():
            st.subheader("Pre-computed Distillation Results")
            df = pd.read_csv(distill_results_path)
            st.dataframe(df, use_container_width=True)
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
        lambda_w = st.slider("lambda (label weight)", 0.0, 1.0, 0.5, 0.05,
                             key="dist_lambda",
                             help="lambda=1.0 -> only labels, lambda=0.0 -> only rationales")
        epochs = st.slider("Epochs", 1, 10, 3, key="dist_epochs")
        lr = st.select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4, key="dist_lr")
        n_train = st.slider("Training examples", 10, 100, 30, key="dist_ntrain")
        n_test = st.slider("Test examples", 5, 50, 20, key="dist_ntest")
        seed = st.number_input("Seed", value=42, key="dist_seed")

    info = BENCHMARK_REGISTRY[bench_key]
    st.caption(f"**{info['label']}** -- {info['description']}  |  Classes: {', '.join(info['choices'])}")

    # ── Phase 1: Generate rationales ─────────────────────────────────────
    st.subheader("Phase 1 -- Teacher Rationale Generation")
    st.markdown(
        "The teacher LLM produces a **label + natural language rationale** for each training example."
    )

    if st.button("Generate Rationales", key="gen_rat"):
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
                    f"**Label:** {ex.label}  |  **Rationale:** _{ex.rationale}_\n\n---"
                )
    else:
        st.info("Click above to generate rationales from the teacher LLM.")
        return

    # ── Phase 2: Evaluate baseline ───────────────────────────────────────
    st.subheader("Phase 2 -- Baseline (Original Student)")

    if st.button("Evaluate baseline (no distillation)", key="eval_base"):
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
    st.subheader("Phase 3 -- LoRA Fine-Tuning")
    st.markdown(
        f"Multi-task loss with **lambda = {lambda_w}**: "
        f"{lambda_w:.0%} label prediction, {1-lambda_w:.0%} rationale generation."
    )

    if st.button("Distill Student", type="primary", key="distill_btn"):
        from distill import DistillDataset, fine_tune_student

        task = st.session_state["dist_task"]
        annotated = st.session_state["dist_annotated"]

        train_annotated = annotated

        with st.status("Preparing dataset...", expanded=True) as status:
            st.write("Tokenizing multi-task training data...")
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
    st.subheader("Phase 4 -- Evaluate Distilled Student")

    if st.button("Evaluate distilled model", key="eval_dist"):
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
            title=f"Distillation Result (lambda={lambda_w})",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Loss function details"):
            st.markdown(
                f"**Multi-task loss:** `L = lambda * L_label + (1-lambda) * L_rationale`\n\n"
                f"- **lambda = {lambda_w}** -> {lambda_w:.0%} of training examples are label-only, "
                f"{1-lambda_w:.0%} are rationale+label\n"
                f"- Implemented via **deterministic stratified dataset mixing**\n"
                f"- **LoRA** adapters fine-tuned (~1-3% of parameters)\n"
                f"- Standard **cross-entropy** on response tokens (prompt tokens masked with -100)"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  YOUR SAVINGS CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════


def page_savings():
    from pricing import LLM_PRICING, SLM_HOSTING, project_monthly, get_model_price

    st.title("Your Savings Calculator")
    st.markdown(
        "**How much could you save?** Estimate the impact of replacing direct LLM API calls "
        "with a hybrid approach: an LLM picks the right examples, a small local model does the work. "
        "Accuracy differences are factored into the real cost."
    )

    # ── Step 0: Optional — test with your own examples ───────────────────
    with st.expander("Optional: measure accuracy on your own examples", expanded=False):
        st.markdown(
            "Paste your own task examples below to get a **measured hybrid accuracy** "
            "instead of a guess. This feeds directly into the cost model."
        )

        if _ML:
            from tasks import build_custom_task, TaskExample
            from slm import generate_answer_generic
            from selector import RandomSelector, GenericLLMSelector

            with st.form("savings_custom_task"):
                sav_task_name = st.text_input("Task name", value="My Task", key="sav_task_name")
                sav_task_instr = st.text_area("Instruction", placeholder="Classify whether this email is spam or legitimate.", key="sav_task_instr")
                sav_task_choices = st.text_input("Answer choices (comma-separated)", placeholder="spam, legitimate", key="sav_task_choices")
                sav_task_examples = st.text_area(
                    "Examples (one per line: `input text ||| label`). Need at least 10 for meaningful results.",
                    height=200,
                    placeholder="Buy now, limited offer!!! ||| spam\nHi, meeting at 3pm tomorrow ||| legitimate\n...",
                    key="sav_task_examples",
                )
                sav_slm = st.selectbox("SLM model", _SLM_OPTIONS[:3], format_func=_short, key="sav_slm")
                sav_provider = st.selectbox("LLM provider", ["openai", "anthropic"], key="sav_test_provider")
                sav_llm = st.text_input("LLM model", value="gpt-4o-mini" if sav_provider == "openai" else "claude-haiku-4-5", key="sav_test_llm")
                sav_api_key = st.text_input("API key", type="password", key="sav_test_key")
                sav_submitted = st.form_submit_button("Run quick benchmark on my examples")

            if sav_submitted and sav_task_instr and sav_task_choices and sav_task_examples:
                choices = [c.strip() for c in sav_task_choices.split(",") if c.strip()]
                try:
                    task = build_custom_task(sav_task_name, sav_task_instr, choices,
                                             sav_task_examples, train_ratio=0.6, seed=42)
                except ValueError as e:
                    st.error(str(e))
                    task = None

                if task and len(task.test_set) >= 2:
                    model, tokenizer, device = _load_slm(sav_slm)
                    n = len(task.test_set)
                    progress = st.progress(0, text="Evaluating...")

                    zs_correct, llm_correct, rand_correct = 0, 0, 0
                    random_sel = RandomSelector()
                    llm_sel = None
                    if sav_api_key:
                        llm_sel = GenericLLMSelector(
                            sav_provider, sav_llm, sav_api_key,
                            task_instruction=task.instruction, task_choices=task.choices,
                            cache_dir=CACHE_DIR / "selector",
                        )

                    for i, ex in enumerate(task.test_set):
                        progress.progress((i + 1) / n, text=f"Question {i+1}/{n}...")
                        # Zero-shot
                        ans, _, _ = generate_answer_generic(
                            model, tokenizer, device,
                            task.instruction, task.choices, ex.input_text, ex.context, [],
                        )
                        zs_correct += int(ans == ex.correct)
                        # Random
                        shots = random_sel.select(task.train_pool, ex, 3, seed=42 + i)
                        ans, _, _ = generate_answer_generic(
                            model, tokenizer, device,
                            task.instruction, task.choices, ex.input_text, ex.context, shots,
                        )
                        rand_correct += int(ans == ex.correct)
                        # LLM-assisted
                        if llm_sel:
                            shots = llm_sel.select(task.train_pool, ex, 3, seed=42 + i)
                            ans, _, _ = generate_answer_generic(
                                model, tokenizer, device,
                                task.instruction, task.choices, ex.input_text, ex.context, shots,
                            )
                            llm_correct += int(ans == ex.correct)

                    progress.progress(1.0, text="Done!")

                    zs_acc = zs_correct / n
                    rand_acc = rand_correct / n
                    llm_acc = llm_correct / n if llm_sel else rand_acc

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Zero-Shot", f"{zs_acc:.0%}")
                    r2.metric("Random", f"{rand_acc:.0%}")
                    r3.metric("LLM-Assisted", f"{llm_acc:.0%}")

                    best_hybrid = max(rand_acc, llm_acc)
                    st.session_state["sav_measured_acc"] = best_hybrid
                    st.success(
                        f"Measured best hybrid accuracy: **{best_hybrid:.0%}** "
                        f"on {n} test examples. This is now used in the cost model below."
                    )
        else:
            st.info(
                "Install inference dependencies to test with your own examples: "
                "`pip install torch transformers datasets accelerate openai anthropic`"
            )

    # ── Step 1: Your current LLM setup ───────────────────────────────────
    st.subheader("1. Your current LLM setup")
    c1, c2 = st.columns(2)
    direct_model = c1.selectbox(
        "Which LLM do you use today?",
        list(LLM_PRICING.keys()),
        index=1,
        key="sav_direct",
    )
    price = get_model_price(direct_model)
    c2.caption(
        f"**Pricing:** ${price['input']:.2f} / 1M input tokens, "
        f"${price['output']:.2f} / 1M output tokens"
    )

    t1, t2 = st.columns(2)
    direct_in = t1.number_input(
        "Avg input tokens per request",
        min_value=10, max_value=50_000, value=500, step=50,
        key="sav_in",
        help="Includes system prompt + user message. Check your LLM dashboard for actuals.",
    )
    direct_out = t2.number_input(
        "Avg output tokens per request",
        min_value=1, max_value=10_000, value=50, step=10,
        key="sav_out",
        help="Length of the model's response. For classification tasks this is typically 1-20.",
    )

    a1, a2 = st.columns(2)
    direct_acc = a1.slider(
        "Your LLM's accuracy on this task",
        0.5, 1.0, 0.90, 0.01,
        key="sav_acc",
        help="If you don't know, 85-95% is typical for GPT-4o on classification.",
    )
    monthly_vol = a2.number_input(
        "Monthly request volume",
        min_value=100, max_value=10_000_000, value=100_000, step=10_000,
        key="sav_vol",
    )

    current_monthly = monthly_vol * (
        direct_in * price["input"] / 1e6
        + direct_out * price["output"] / 1e6
    )
    st.metric("Your current monthly LLM spend", f"${current_monthly:,.2f}")

    st.divider()

    # ── Step 2: Hybrid configuration ─────────────────────────────────────
    st.subheader("2. Hybrid configuration")

    h1, h2 = st.columns(2)
    selector_model = h1.selectbox(
        "LLM selector model (picks examples for the SLM)",
        list(LLM_PRICING.keys()),
        index=1,
        key="sav_selector",
    )
    sel_price = get_model_price(selector_model)
    h2.caption(
        f"**Pricing:** ${sel_price['input']:.2f} / 1M in, "
        f"${sel_price['output']:.2f} / 1M out"
    )

    s1, s2 = st.columns(2)
    sel_in = s1.number_input(
        "Selector input tokens/question (estimated)",
        min_value=50, max_value=10_000, value=800, step=100,
        key="sav_sel_in",
        help="~800 tokens for a typical few-shot selection prompt.",
    )
    sel_out = s2.number_input(
        "Selector output tokens/question (estimated)",
        min_value=5, max_value=1_000, value=60, step=10,
        key="sav_sel_out",
        help="~60 tokens for the selector's response (JSON indices).",
    )

    # Use measured accuracy if available from the custom benchmark
    measured_acc = st.session_state.get("sav_measured_acc")
    default_hybrid_acc = measured_acc if measured_acc else 0.65
    hybrid_acc_help = (
        "Measured from your examples above."
        if measured_acc
        else "Run a benchmark to measure this. Typical: 55-70% for small models with good example selection."
    )
    hybrid_acc = st.slider(
        "Expected hybrid accuracy",
        0.3, 1.0, default_hybrid_acc, 0.01,
        key="sav_hybrid_acc",
        help=hybrid_acc_help,
    )

    hosting_keys = [k for k in SLM_HOSTING if k != "custom"] + ["custom"]
    hosting_key = st.selectbox(
        "SLM hosting",
        hosting_keys,
        format_func=lambda k: SLM_HOSTING[k]["label"],
        key="sav_hosting",
    )
    custom_hosting = 0.0
    if hosting_key == "custom":
        custom_hosting = st.number_input(
            "Monthly hosting cost ($)", 0.0, 10_000.0, 50.0, 10.0,
            key="sav_custom_host",
        )

    st.divider()

    # ── Step 3: Results ──────────────────────────────────────────────────
    st.subheader("3. Results")

    proj = project_monthly(
        monthly_vol,
        direct_model=direct_model,
        direct_input_tpq=float(direct_in),
        direct_output_tpq=float(direct_out),
        direct_accuracy=direct_acc,
        selector_model=selector_model,
        selector_input_tpq=float(sel_in),
        selector_output_tpq=float(sel_out),
        slm_hosting_key=hosting_key,
        custom_hosting_cost=custom_hosting,
        hybrid_accuracy=hybrid_acc,
    )

    # ── Primary view: accuracy-adjusted ──────────────────────────────────
    acc_delta = hybrid_acc - direct_acc

    view = st.radio(
        "Cost view",
        ["Raw cost", "Accuracy-adjusted", "With error routing"],
        horizontal=True,
        help=(
            "**Raw cost**: total spend ignoring accuracy differences. "
            "**Accuracy-adjusted**: cost per correct answer, normalized. "
            "**With error routing**: assumes hybrid misses are re-sent to the LLM."
        ),
    )

    if view == "Raw cost":
        h_cost = proj.hybrid_total
        h_savings = proj.savings_abs
        h_pct = proj.savings_pct
        view_note = ""
    elif view == "Accuracy-adjusted":
        h_cost = proj.hybrid_total  # raw cost is the same, but we show per-correct
        h_savings = proj.adjusted_savings_abs
        h_pct = proj.adjusted_savings_pct
        view_note = f"Comparing cost to produce **{int(monthly_vol * direct_acc):,}** correct answers"
    else:  # With error routing
        h_cost = proj.hybrid_total_with_errors
        h_savings = proj.savings_with_errors_abs
        h_pct = proj.savings_with_errors_pct
        view_note = (
            f"Assumes {(1 - hybrid_acc):.0%} of hybrid answers are wrong and get "
            f"re-routed to {direct_model} (+${proj.error_routing_cost:,.2f}/mo)"
        )

    if view_note:
        st.caption(view_note)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current spend", f"${proj.pure_llm_cost:,.2f}/mo")
    m2.metric("Hybrid cost", f"${h_cost:,.2f}/mo")
    savings_sign = "+" if h_savings >= 0 else ""
    m3.metric("Monthly savings", f"${h_savings:,.2f}", f"{h_pct:.0f}%")
    m4.metric("Annual savings", f"${h_savings * 12:,.0f}")

    import plotly.graph_objects as go

    fig = go.Figure()
    bars_x = ["Current (pure LLM)"]
    bars_y = [proj.pure_llm_cost]
    bars_color = ["#ef4444"]
    bars_text = [f"${proj.pure_llm_cost:,.2f}"]

    if view == "With error routing" and proj.error_routing_cost > 0:
        # Stacked bar: hybrid base + error routing cost
        fig.add_trace(go.Bar(
            x=["Hybrid (David vs Goliath)"], y=[proj.hybrid_total],
            name="Hybrid base", marker_color="#10b981",
            text=[f"${proj.hybrid_total:,.2f}"], textposition="inside",
        ))
        fig.add_trace(go.Bar(
            x=["Hybrid (David vs Goliath)"], y=[proj.error_routing_cost],
            name="Error routing", marker_color="#f59e0b",
            text=[f"+${proj.error_routing_cost:,.2f}"], textposition="inside",
        ))
        fig.add_trace(go.Bar(
            x=["Current (pure LLM)"], y=[proj.pure_llm_cost],
            name="Pure LLM", marker_color="#ef4444",
            text=[f"${proj.pure_llm_cost:,.2f}"], textposition="outside",
        ))
        fig.update_layout(barmode="stack")
    else:
        fig.add_trace(go.Bar(
            x=["Current (pure LLM)", "Hybrid (David vs Goliath)"],
            y=[proj.pure_llm_cost, h_cost],
            marker_color=["#ef4444", "#10b981"],
            text=[f"${proj.pure_llm_cost:,.2f}", f"${h_cost:,.2f}"],
            textposition="outside",
        ))

    fig.update_layout(
        template="plotly_white", height=380,
        yaxis_title="Monthly Cost ($)", title="Cost Comparison",
        showlegend=(view == "With error routing"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Accuracy-adjusted detail ─────────────────────────────────────────
    st.subheader("4. The accuracy picture")

    if acc_delta < 0:
        st.markdown(
            f"The hybrid is **{abs(acc_delta):.0%} less accurate** ({hybrid_acc:.0%} vs {direct_acc:.0%}). "
            f"That accuracy gap has a real cost:"
        )
    elif acc_delta == 0:
        st.markdown(f"Both approaches have the same accuracy ({direct_acc:.0%}).")
    else:
        st.success(
            f"The hybrid is **{acc_delta:.0%} more accurate** AND cheaper. No trade-off needed."
        )

    acc_c1, acc_c2, acc_c3 = st.columns(3)
    acc_c1.metric(f"Cost per 1K correct ({direct_model})",
                  f"${proj.direct_cost_per_correct:,.4f}")
    acc_c2.metric("Cost per 1K correct (hybrid)",
                  f"${proj.hybrid_cost_per_correct:,.4f}")
    if proj.hybrid_cost_per_correct > 0:
        cpc_ratio = proj.direct_cost_per_correct / proj.hybrid_cost_per_correct
        acc_c3.metric("Efficiency ratio", f"{cpc_ratio:.1f}x",
                      "hybrid is cheaper per correct answer" if cpc_ratio > 1 else "LLM is cheaper per correct answer")

    if acc_delta < 0:
        st.markdown("##### What happens to the wrong answers?")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"**Option A: Accept errors**\n\n"
                f"- Monthly errors: **{int(monthly_vol * (1 - hybrid_acc)):,}**\n"
                f"- Raw savings: **${proj.savings_abs:,.2f}/mo**\n"
                f"- You save money but lose accuracy"
            )
        with col_b:
            st.markdown(
                f"**Option B: Route errors to LLM**\n\n"
                f"- Re-process {(1 - hybrid_acc):.0%} of calls via {direct_model}\n"
                f"- Error routing cost: **${proj.error_routing_cost:,.2f}/mo**\n"
                f"- Net savings: **${proj.savings_with_errors_abs:,.2f}/mo ({proj.savings_with_errors_pct:.0f}%)**"
            )
        if proj.savings_with_errors_abs > 0:
            st.info(
                f"Even routing all errors back to {direct_model}, "
                f"you still save **${proj.savings_with_errors_abs:,.2f}/mo**."
            )
        elif proj.savings_with_errors_abs <= 0:
            st.warning(
                f"With error routing, the hybrid is **more expensive** by "
                f"${abs(proj.savings_with_errors_abs):,.2f}/mo. "
                f"You need higher hybrid accuracy or a cheaper selector to make this work."
            )

    with st.expander("Full cost breakdown"):
        st.markdown(
            f"| | Per Question | Monthly ({monthly_vol:,} calls) |\n"
            f"|---|---|---|\n"
            f"| **Current: {direct_model}** | {direct_in} in + {direct_out} out tokens | "
            f"**${proj.pure_llm_cost:,.2f}** |\n"
            f"| Hybrid: Selector ({selector_model}) | {sel_in} in + {sel_out} out tokens | "
            f"${proj.hybrid_llm_cost:,.2f} |\n"
            f"| Hybrid: SLM hosting | -- | ${proj.hybrid_hosting_cost:,.0f} |\n"
            f"| **Hybrid total (raw)** | | **${proj.hybrid_total:,.2f}** |\n"
            f"| Error routing ({(1 - hybrid_acc):.0%} of {monthly_vol:,}) | | "
            f"${proj.error_routing_cost:,.2f} |\n"
            f"| **Hybrid total (with error routing)** | | "
            f"**${proj.hybrid_total_with_errors:,.2f}** |"
        )
        if proj.break_even_volume > 0:
            st.info(f"Break-even at **{proj.break_even_volume:,}** calls/month to cover hosting costs.")


# ── Route ────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    page_dashboard()
elif page == "Live Demo":
    page_live_demo()
elif page == "Benchmark Lab":
    page_benchmark_lab()
elif page == "Distillation":
    page_distillation()
else:
    page_savings()
