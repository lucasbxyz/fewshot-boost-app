# David vs Goliath

Benchmark + toolkit: small language models (SLMs) vs LLM-assisted few-shot selection and knowledge distillation.

## Key commands

```bash
pip install -e ".[all]"       # full install (torch, transformers, peft, openai, anthropic, streamlit)
streamlit run app.py           # web dashboard at localhost:8501
dvg                            # CLI interactive wizard
dvg run --help                 # benchmark runner
dvg cost --help                # savings calculator
dvg distill --help             # distillation pipeline
dvg results                    # show pre-computed results
```

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app — 5 tabs: Dashboard, Live Demo, Benchmark Lab, Distillation, Your Savings |
| `cli.py` | Typer CLI — commands: run, demo, distill, cost, results + interactive wizard |
| `pricing.py` | Token-based cost estimation engine (all assumptions user-configurable) |
| `distill.py` | Distilling Step-by-Step pipeline: teacher rationales → LoRA fine-tuning of student |
| `tasks.py` | Generic Task/TaskExample abstraction + loaders for BoolQ, SST-2, AG News |
| `slm.py` | SLM model loading + inference (HuggingFace, supports MPS/CUDA/CPU) |
| `selector.py` | Few-shot selectors: RandomSelector + GenericLLMSelector (OpenAI/Anthropic) |
| `benchmark.py` | Legacy BoolQ loader (kept for backward compat, prefer tasks.py) |
| `data/` | Pre-computed benchmark CSVs (summary, per-question, per-category) |

## Stack

Python 3.10+, Streamlit, Plotly, Typer, Rich, InquirerPy, HuggingFace Transformers, PEFT (LoRA), OpenAI SDK, Anthropic SDK

## Important patterns

- `app.py` checks `_ML` flag — dashboard works without torch, inference tabs show install hint
- Cost estimation is fully token-based — `pricing.project_monthly()` takes measured or estimated tokens per question
- `CostEstimate` dataclass captures measured token counts from benchmark runs
- Distillation only annotates `train_pool` (not test set) to prevent data leakage
- `selector.py` guards `rng.sample` with `min(k, len(train_pool))`
- CLI entry point is `dvg` (defined in pyproject.toml `[project.scripts]`)

## Repo

https://github.com/lucasbxyz/fewshot-boost-app
