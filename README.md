---
title: David vs Goliath
emoji: "\U0001F3AF"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# David vs Goliath

**Cut your LLM API costs by up to 90% without sacrificing accuracy.**

Instead of sending every request to an expensive LLM, David vs Goliath uses a large model only to *select the right examples*, then a tiny local model (135M-1.7B parameters) does the actual work. The result: comparable accuracy at a fraction of the cost.

## The Problem

You're spending $500-5,000/month on LLM API calls for classification, Q&A, or labeling tasks. Every single request hits GPT-4o or Claude -- even though 80% of those requests are routine and predictable.

## The Solution

```
┌──────────────┐                    ┌────────────────┐
│  Your Query  │ ──── LLM picks ──> │  Best Examples │
└──────────────┘    3 examples      └───────┬────────┘
                                            │
                                    ┌───────▼────────┐
                                    │  Small Model   │
                                    │  (135M params) │
                                    │  answers fast  │
                                    │  & free (local)│
                                    └───────┬────────┘
                                            │
                                        answer
```

**Three strategies benchmarked:**
1. **Zero-Shot** -- small model answers cold (baseline)
2. **Random Few-Shot** -- randomly picked examples
3. **LLM-Assisted** -- LLM picks the *best* examples for each question

**Plus distillation:** the LLM teaches the small model to *reason*, not just classify -- via LoRA fine-tuning with rationale generation.

## Key Results

| Strategy | SmolLM2-135M | SmolLM2-360M |
|---|---|---|
| Zero-Shot | 49-52% | 54-57% |
| Random Few-Shot | 45-52% | 50-54% |
| **LLM-Assisted** | **50-63%** | **53-60%** |

**Best boost: +14.2 percentage points** over zero-shot baseline.

Over **20,000 predictions** across 3 model sizes, 3 LLM selectors (GPT-4o, GPT-4o-mini, Claude Haiku), 4 seeds, and 5 runs per configuration.

## Cost Impact

| Scenario | Pure LLM | Hybrid (DvG) | Savings |
|---|---|---|---|
| 100K calls/mo, GPT-4o | $2,750/mo | $156/mo | **$2,594/mo (94%)** |
| 100K calls/mo, GPT-4o-mini | $27/mo | $16/mo | **$11/mo (42%)** |
| 1M calls/mo, GPT-4o | $27,500/mo | $1,560/mo | **$25,940/mo (94%)** |

*Self-hosted SLM. Actual savings depend on your token usage and hosting choice.*

## Quickstart

### Dashboard (works everywhere, no GPU needed)

```bash
git clone https://github.com/lucasbxyz/fewshot-boost-app.git
cd fewshot-boost-app
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501** with 5 tabs: Dashboard, Live Demo, Benchmark Lab, Distillation, and Your Savings calculator.

### CLI Tool

```bash
pip install -e ".[all]"    # installs everything including the dvg command
dvg                        # interactive wizard
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

### Enable Live Inference

Live Demo, Benchmark Lab, Distillation, and CLI commands that run models need:

```bash
pip install torch transformers datasets accelerate anthropic openai peft
```

Or: `pip install -e ".[inference,llm]"`

## Deploy to Streamlit Community Cloud (Free)

The dashboard works on Streamlit Cloud with zero configuration. Live inference features (Demo, Benchmark Lab, Distillation) require local hardware.

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Select this repo, branch `main`, main file `app.py`
4. Deploy -- live within minutes

## Supported Benchmarks

| Benchmark | Task | Classes |
|---|---|---|
| **BoolQ** | Yes/no comprehension | yes, no |
| **SST-2** | Sentiment analysis | positive, negative |
| **AG News** | Topic classification | World, Sports, Business, Sci/Tech |
| **Custom** | Define your own task template | any |

## Architecture

```
david-vs-goliath/
├── app.py              # Streamlit dashboard (5 tabs)
├── cli.py              # CLI tool (interactive wizard + commands)
├── selector.py         # Few-shot selectors (random + LLM-assisted w/ caching)
├── slm.py              # Small model inference (CUDA / MPS / CPU)
├── tasks.py            # Task abstraction + benchmark loaders
├── distill.py          # Distilling Step-by-Step (LoRA fine-tuning)
├── pricing.py          # Token-based cost estimation engine
├── benchmark.py        # Legacy BoolQ loader (backward compat)
├── config.yaml         # Configuration
├── tests/              # Test suite (41 tests)
├── data/               # Pre-computed benchmark CSVs
├── cache/              # LLM selector response cache (gitignored)
└── results/            # Persisted run results (gitignored)
```

## Tech Stack

- **Web UI**: Streamlit, Plotly
- **CLI**: Typer, Rich, InquirerPy
- **Models**: HuggingFace Transformers (SmolLM2, TinyLlama, Phi-3, Gemma)
- **Distillation**: LoRA via PEFT, multi-task training
- **LLM Selectors**: OpenAI API, Anthropic API (with token tracking + caching)
- **Benchmarks**: Google BoolQ, Stanford SST-2, AG News (via HuggingFace Datasets)
- **Cost Estimation**: Token-based LLM pricing + SLM hosting tiers

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```
