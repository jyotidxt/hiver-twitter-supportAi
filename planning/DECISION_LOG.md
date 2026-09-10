# Decision Log

This document tracks key decisions made during development.

---

## Phase 1 — Planning

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| 1 | — | Single-brand focus | Simplifies data collection, enables brand-specific language modeling, and allows controlled evaluation. |
| 2 | — | Kaggle dataset selection | "Customer Support on Twitter" by ThoughtVector provides ~1.6M tweets across multiple brands with reply chain metadata. |
| 3 | — | English-only pipeline | Reduces complexity; the dataset is predominantly English. |

## Phase 2 — Data Engineering & Analytics

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| 4 | — | AmazonHelp as default brand | High conversation volume, diverse issue set (e-commerce covers orders, returns, billing, technical). Configurable in `configs/config.yaml`. |
| 5 | — | YAML configuration over hardcoded values | Enables reproducible experiments with different parameters without code changes. |
| 6 | — | BFS thread reconstruction | Handles branching conversations where multiple replies stem from a single tweet. Depth-limited to prevent infinite loops. |
| 7 | — | Preserve emojis and meaningful punctuation | Customer tone carries sentiment signal; aggressive cleaning would destroy useful features for downstream tasks. |
| 8 | — | TF-IDF + K-Means for intent discovery | Lightweight, interpretable, no GPU required. Provides a starting point for humans to define the final intent taxonomy. |
| 9 | — | Matplotlib-only visualizations | Ensures reproducibility without complex dependencies. Seaborn used only for styling, not plot logic. |
| 10 | — | Separate `annotation_ready.csv` format | One row per conversation with pre-formatted dialogue simplifies the manual annotation workflow. |
| 11 | — | Tests with pytest | Standard Python testing framework; tests cover data loading, cleaning, threading, and config. |

---

## Phase 3 — Intent Taxonomy & Golden Dataset

| # | Date | Decision | Rationale |
|---|------|----------|-----------|
| 12 | — | 12-intent taxonomy across 6 categories | Balances coverage with annotation consistency. More than 15 intents would fragment the data; fewer than 10 would lack routing granularity. |
| 13 | — | Brand-derived intents (not generic) | Taxonomy intents come from Phase 2 EDA cluster analysis and AmazonHelp conversation patterns, not from generic benchmarks like Banking77. |
| 14 | — | Single primary intent per message | Avoids multi-label complexity. Secondary signals (urgency, emotion) are captured through escalation labels, not duplicate intents. |
| 15 | — | Root-cause priority for ambiguous messages | When multiple complaints exist, the intent that resolves the root cause is selected. This aligns with actual support workflows. |
| 16 | — | Decision tree for annotation | Mermaid flowchart provides deterministic path to reduce inter-annotator disagreement on borderline cases. |
| 17 | — | Versioned taxonomy with deprecation rules | New intents can be added via MINOR version increments without invalidating previously labeled golden datasets. |

---

*Updated as decisions are made throughout the project.*
