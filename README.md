# ⚡ FewShot Boost

**Making tiny language models smarter with intelligent few-shot example selection.**

A small language model (135M–360M parameters) answers yes/no questions barely better than a coin flip. But what if a large language model could pick the *right* examples to show it first?

FewShot Boost benchmarks three prompting strategies on multiple datasets and provides an interactive dashboard, live inference demo, benchmark lab with cost estimation, and a pip-installable CLI.

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
git clone https://github.com/YOUR_USERNAME/fewshot-boost-app.git
cd fewshot-boost-app
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501** with three tabs: Dashboard, Live Demo, and Benchmark Lab.

### Option B: CLI Tool

```bash
pip install -e ".[all]"    # installs everything including the fewshot-boost command
fewshot-boost              # launches interactive wizard
```

Or use flags directly:

```bash
# Run a benchmark
fewshot-boost run --slm HuggingFaceTB/SmolLM2-135M-Instruct --benchmark boolq --api-key $OPENAI_API_KEY

# Quick single-question demo
fewshot-boost demo --api-key $OPENAI_API_KEY

# Cost estimation calculator
fewshot-boost cost --llm-model gpt-4o-mini --monthly-volume 100000

# View pre-computed results
fewshot-boost results
```

### Enable Live Inference (both web & CLI)

Live Demo, Benchmark Lab, and CLI commands that run models need:

```bash
pip install torch transformers datasets accelerate anthropic openai
```

Or via optional extras: `pip install -e ".[inference,llm]"`

## Deploy to Streamlit Community Cloud (Free)

The dashboard works on Streamlit Cloud with zero configuration. Live inference is local-only.

### Step-by-step

1. **Push to GitHub** — create a new repo and push this folder:
   ```bash
   cd fewshot-boost-app
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/fewshot-boost-app.git
   git branch -M main
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. **Click "New app"** and select:
   - Repository: `YOUR_USERNAME/fewshot-boost-app`
   - Branch: `main`
   - Main file path: `app.py`

4. **Click "Deploy"** — your app will be live at `https://your-app-name.streamlit.app` within a few minutes.

That's it. The app auto-redeploys whenever you push to `main`.

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

The benchmark runs each strategy across multiple seeds and runs, then computes accuracy with confidence intervals.

## Project Structure

```
fewshot-boost-app/
├── app.py              # Streamlit app (dashboard + live demo + benchmark lab)
├── cli.py              # CLI tool (interactive wizard + flag commands)
├── tasks.py            # Generic task abstraction + benchmark loaders
├── pricing.py          # LLM pricing tables + cost calculator
├── benchmark.py        # BoolQ dataset loading (legacy)
├── slm.py              # Small model inference (original + generic)
├── selector.py         # Few-shot selectors (original + generic w/ token tracking)
├── evaluator.py        # Accuracy & statistics
├── config.yaml         # Configuration (placeholder API keys)
├── pyproject.toml      # pip install -e . (makes fewshot-boost command available)
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
- **LLM Selectors**: OpenAI API, Anthropic API (with token tracking)
- **Benchmarks**: Google BoolQ, Stanford SST-2, AG News (via HuggingFace Datasets)
- **Cost Estimation**: Built-in LLM pricing tables + SLM hosting tiers
