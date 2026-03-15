# ⚔️ David vs Goliath

**When tiny models outsmart the giants.**

A small language model (135M–360M parameters) answers yes/no questions barely better than a coin flip. But what if a large language model could pick the *right* examples to show it first — or even teach it to reason?

David vs Goliath benchmarks three prompting strategies on multiple datasets, distills reasoning from teacher LLMs into student SLMs via LoRA, and provides an interactive dashboard, live inference demo, benchmark lab with cost estimation, and a pip-installable CLI.

## Key Results

| Strategy | SmolLM2-135M | SmolLM2-360M |
|---|---|---|
| Zero-Shot | 49–52% | 54–57% |
| Random Few-Shot | 45–52% | 50–54% |
| **LLM-Assisted** | **50–63%** | **53–60%** |

**Best boost: +14.2 percentage points** over zero-shot (GPT-4o-mini selector, seed 789).

Over 20,000 predictions across 3 model sizes, 3 LLM selectors (GPT-4o, GPT-4o-mini, Claude Haiku), 4 seeds, and 5 runs per configuration.

## Quickstart

### Option A: Web Dashboard (Streamlit)

```bash
git clone https://github.com/lucasbxyz/fewshot-boost-app.git
cd fewshot-boost-app
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501** with tabs: Dashboard, Live Demo, Benchmark Lab, and Distillation.

### Option B: CLI Tool

```bash
pip install -e ".[all]"    # installs everything including the dvg command
dvg                        # launches interactive wizard
```

Or use flags directly:

```bash
# Run a benchmark
dvg run --slm HuggingFaceTB/SmolLM2-135M-Instruct --benchmark boolq --api-key $OPENAI_API_KEY

# Quick single-question demo
dvg demo --api-key $OPENAI_API_KEY

# Cost estimation calculator
dvg cost --llm-model gpt-4o-mini --monthly-volume 100000

# Distill reasoning from teacher to student
dvg distill --benchmark boolq --llm-model gpt-4o-mini --api-key $OPENAI_API_KEY

# View pre-computed results
dvg results
```

### Enable Live Inference (both web & CLI)

Live Demo, Benchmark Lab, Distillation, and CLI commands that run models need:

```bash
pip install torch transformers datasets accelerate anthropic openai peft
```

Or via optional extras: `pip install -e ".[inference,llm]"`

## Deploy to Streamlit Community Cloud (Free)

The dashboard works on Streamlit Cloud with zero configuration. Live inference is local-only.

1. **Push to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. **Click "New app"** and select:
   - Repository: `lucasbxyz/fewshot-boost-app`
   - Branch: `main`
   - Main file path: `app.py`
4. **Click "Deploy"** — live within minutes. Auto-redeploys on push to `main`.

## How It Works

```
┌─────────────────────────────────────────────────┐
│                  Test Question                   │
│  "Is the capital of France Paris?"               │
└──────────────────────┬──────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌───────────┐
   │ Zero-Shot│  │  Random  │  │LLM-Assisted│
   │ (no ex.) │  │ (random) │  │ (smart)    │
   └────┬─────┘  └────┬─────┘  └─────┬─────┘
        │              │              │
        ▼              ▼              ▼
   ┌──────────────────────────────────────┐
   │         Small Language Model          │
   │    (SmolLM2-135M / 360M / 1.7B)     │
   └──────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
     answer          answer        answer
```

1. **Zero-Shot** — the SLM answers with no examples (baseline)
2. **Random** — *k* examples are randomly sampled from a training pool
3. **LLM-Assisted** — a large model (GPT-4o, Claude, etc.) picks the *k* most relevant examples
4. **Distillation** — the teacher LLM generates labels + rationales, then the student SLM is LoRA-fine-tuned with multi-task loss

## Project Structure

```
david-vs-goliath/
├── app.py              # Streamlit app (dashboard + live demo + benchmark lab + distillation)
├── cli.py              # CLI tool (interactive wizard + flag commands)
├── distill.py          # Distilling Step-by-Step pipeline (LoRA fine-tuning)
├── tasks.py            # Generic task abstraction + benchmark loaders
├── pricing.py          # LLM pricing tables + cost calculator
├── slm.py              # Small model inference
├── selector.py         # Few-shot selectors (random + LLM-assisted w/ token tracking)
├── config.yaml         # Configuration (placeholder API keys)
├── pyproject.toml      # pip install -e . (makes dvg command available)
├── requirements.txt    # Python dependencies
└── data/
    ├── all_results_summary.csv
    ├── all_results_per_question.csv
    └── all_results_per_category.csv
```

## Supported Benchmarks

| Benchmark | Task | Classes |
|---|---|---|
| **BoolQ** | Yes/no comprehension | yes, no |
| **SST-2** | Sentiment analysis | positive, negative |
| **AG News** | Topic classification | World, Sports, Business, Sci/Tech |
| **Custom** | Define your own task template | any |

## Tech Stack

- **Web UI**: Streamlit, Plotly
- **CLI**: Typer, Rich, InquirerPy
- **Models**: HuggingFace Transformers (SmolLM2, TinyLlama, Phi-3, Gemma)
- **Distillation**: LoRA via PEFT, multi-task training
- **LLM Selectors**: OpenAI API, Anthropic API (with token tracking)
- **Benchmarks**: Google BoolQ, Stanford SST-2, AG News (via HuggingFace Datasets)
- **Cost Estimation**: Token-based LLM pricing + SLM hosting tiers
