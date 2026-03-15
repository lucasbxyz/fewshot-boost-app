# David vs Goliath

Cut your LLM API costs by up to 90% — intelligent few-shot selection and knowledge distillation.

## Key commands

```bash
pip install -e ".[all]"       # full install (torch, transformers, peft, openai, anthropic, streamlit)
streamlit run app.py           # web dashboard at localhost:8501
dvg                            # CLI interactive wizard
dvg run --help                 # benchmark runner
dvg cost --help                # savings calculator
dvg distill --help             # distillation pipeline
dvg results                    # show pre-computed results
python -m pytest tests/ -v     # run test suite (41 tests)
```

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app — 5 tabs: Dashboard, Live Demo, Benchmark Lab, Distillation, Your Savings |
| `cli.py` | Typer CLI — commands: run, demo, distill, cost, results + interactive wizard |
| `pricing.py` | Token-based cost estimation engine (loads from config.yaml, warns if stale) |
| `distill.py` | Distilling Step-by-Step pipeline: teacher rationales → LoRA fine-tuning of student |
| `tasks.py` | Generic Task/TaskExample abstraction + loaders for BoolQ, SST-2, AG News |
| `slm.py` | SLM model loading + inference (HuggingFace, supports CUDA/MPS/CPU) |
| `selector.py` | GenericLLMSelector with token tracking, disk caching, and retry logic |
| `benchmark.py` | Legacy BoolQ loader (kept for backward compat, prefer tasks.py) |
| `tests/` | Test suite — 41 tests covering parsing, pricing, selection, caching |
| `data/` | Pre-computed benchmark CSVs (summary, per-question, per-category) |
| `cache/` | LLM selector response cache (gitignored) |
| `results/` | Persisted run results (gitignored) |

## Stack

Python 3.10+, Streamlit, Plotly, Typer, Rich, InquirerPy, HuggingFace Transformers, PEFT (LoRA), OpenAI SDK, Anthropic SDK

## Important patterns

- `app.py` checks `_ML` flag — dashboard works without torch, inference tabs show install hint
- Cost estimation is fully token-based — `pricing.project_monthly()` takes measured or estimated tokens per question
- Pricing loads from config.yaml if present, warns if data is >6 months stale
- `CostEstimate` dataclass captures measured token counts from benchmark runs
- Distillation only annotates `train_pool` (not test set) to prevent data leakage
- Distillation uses deterministic stratified splitting (not random coin) for λ-weighting
- `selector.py` guards `rng.sample` with `min(k, len(train_pool))`
- `GenericLLMSelector` supports disk caching via `cache_dir` parameter
- CLI persists results to `results/run_history.csv` after each benchmark run
- CLI entry point is `dvg` (defined in pyproject.toml `[project.scripts]`)
- `slm.py` auto-detects CUDA → MPS → CPU; uses float16 on CUDA, float32 elsewhere

## Repo

https://github.com/lucasbxyz/fewshot-boost-app
