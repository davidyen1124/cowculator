<div align="center">

# COWCULATOR 🐄➗📈
Because nothing says “leadership” like forecasting burrito budgets with AI.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Build](https://img.shields.io/badge/CI-GitHub%20Actions-2088ff?logo=github-actions&logoColor=white)](#)
[![Ruff](https://img.shields.io/badge/Lint-ruff-46a2f1.svg)](#)
[![Black](https://img.shields.io/badge/Style-black-000000.svg)](#)
[![mypy](https://img.shields.io/badge/Typing-mypy-2A6DB2.svg)](#)
[![UV](https://img.shields.io/badge/Deps-uv-ff69b4.svg)](#)
[![Conventional Commits](https://img.shields.io/badge/Commits-conventional-ec7600.svg)](#)
[![Commitizen](https://img.shields.io/badge/Commitizen-friendly-blue.svg)](#)
[![Semantic Release](https://img.shields.io/badge/Release-semantic-39c0ba.svg)](#)
[![Pre‑commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](#)
[![Open in VS Code](https://img.shields.io/badge/Open%20in-VS%20Code-007ACC?logo=visual-studio-code)](#)
[![Made With Love](https://img.shields.io/badge/Made%20with-%F0%9F%92%96-lightgrey.svg)](#)
[![No Notebooks](https://img.shields.io/badge/Jupyter-not%20today!-purple.svg)](#)
[![Works on My Machine](https://img.shields.io/badge/Works%20on-My%20Machine-success.svg)](#)
[![AI Inside](https://img.shields.io/badge/AI-inside-8A2BE2.svg)](#)
[![Buzzwords](https://img.shields.io/badge/Buzzwords-10x%20%7C%20Synergy%20%7C%20Alignment-ff69b4.svg)](#)
[![Ship It](https://img.shields.io/badge/Ship%20It-🦫-yellow.svg)](#)
[![Blazing Fast](https://img.shields.io/badge/Performance-blazing-orange.svg)](#)
[![Bus Factor](https://img.shields.io/badge/Bus%20Factor-1-red.svg)](#)
[![100% Not Fake](https://img.shields.io/badge/Tests-100%25*-%23ff69b4.svg)](#)
[![Coverage](https://img.shields.io/badge/Coverage-NaN%25-lightgrey.svg)](#)

<sub>Badges are both documentation and personality at this point.</sub>

</div>

Stop Scrolling: I built an AI that predicts your company’s catering costs while you sleep. It’s edge‑ready, vibe‑forward, and CFO‑compatible. This one weird trick could save ~$0–$1,000,000 depending on how loud the demo speakers are.

TL;DR for Executives

- Downloads CaterCow order survey data, turns it into grown‑up tables, and trains not one but TWO models.
- Exports a tiny JSON bundle so the browser can forecast the next 5 business days without a backend. Yes, really.
- Comes with a web page that makes your ops updates look like a product launch.

Quickstart (Become a Thought Leader in 60 Seconds)

- Install deps: `uv sync`
- Fetch data: `uv run python main.py fetch`
- Train models: `uv run python main.py train`
- Predict tomorrow: `uv run python main.py predict-next`
- Export for the web: `uv run python main.py export-edge` then open `web/index.html`

What You Get (Deliverables You Can Screenshot)

- Raw JSON: `data/raw/order_surveys_*.json`
- Flattened parquet: `data/processed/orders.parquet`
- Artifacts: `artifacts/model.joblib`, `artifacts/metrics.json`, `artifacts/weekday_stats.parquet`, `artifacts/daily_model.joblib`, `artifacts/daily_history.parquet`
- Frontend bundle: `web/edge_bundle.json` + a demo page at `web/index.html`

Architecture, But Make It Inspirational

- Order‑level regression: classic, dependable, scikit‑learn. Predicts per‑order cost like a spreadsheet with self‑esteem.
- Daily time‑series: aggregates to business days, builds lags/rolling means, trains a gradient boosting model. Predicts the future without even asking it nicely.
- Edge export: serializes the tree ensemble to JSON and reenacts it in vanilla JS. It is small. It is fast. It is frankly adorable.

Data → Insight → Bragging: The Flywheel

1) Fetch with `httpx`, write JSONL like a responsible grownup.
2) Flatten with pandas; engineer features that would make 2017 Kaggle proud.
3) Train. Evaluate. Nod thoughtfully at `metrics.json`.
4) Export to `web/`, send a link, take credit.

Live Demo Energy (Locally)

- After export, open `web/index.html`. The page renders a 5‑day forecast grid using only the JSON bundle, a sprinkle of Tailwind, and sheer audacity.

FAQs Nobody Asked But Everyone Needs

- Is this real AI? Yes, in the sense that my laptop gets warm and numbers change.
- Will this replace finance? No, but it will replace awkward silences during standup.
- Can it 10x? It can 10x your confidence and that’s what matters.
- Why no notebooks? Because production is the new prototype.

Repo Tour (You Will Get Asked “Where Is…?”)

- `main.py`: CLI for `fetch`, `train`, `predict-next`, `export-edge`.
- `cowculator/pipeline.py`: data prep, feature engineering, model training, and export logic.
- `data/`: raw and processed files.
- `artifacts/`: models + metrics, aka your receipts.
- `web/`: static UI for flexing to stakeholders.

Install Notes (Bring Your Own Python)

- Python 3.11+
- Uses `uv` for dependency/env management: https://docs.astral.sh/uv/

Roadmap (Definitely Real, Not Aspirational)

- Add “make it go brrr” toggle that increases learning rate by 0.01
- Replace buzzwords with new buzzwords
- Dark mode (for the metrics)

Contributing

- PRs welcome. Memes encouraged. Benchmarks admired. Badges… added.

License

- MIT. See `LICENSE`. For a spicy human summary, see `LICENSE-TLDR.md`.

Footnotes

- `*` Tests are 100%… aspirational. Contributions welcome to make that less of a joke.
