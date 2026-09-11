# Hiver Support AI — Project Report

> **Current Phase**: Phase 3 Complete (Intent Taxonomy & Golden Dataset Annotation System)

---

## Executive Summary

**Hiver Support AI** is a production-quality, reproducible conversational AI customer support analytics and automation system built on the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`), focusing on **AmazonHelp**.

This repository implements the data engineering, exploratory analysis, intent taxonomy design, and golden dataset annotation pipeline necessary to support downstream intent classification, reply generation, and escalation modeling.

---

## Phase Accomplishments

### Phase 1 — Planning & Architecture
- Established system architecture (`planning/00_ARCHITECTURE.md`)
- Conducted multi-brand evaluation and selected **AmazonHelp** (`planning/01_BRAND_SELECTION.md`)
- Defined evaluation label guidelines (`planning/02_LABEL_GUIDELINES.md`)
- Maintained an explicit architecture decision log (`planning/DECISION_LOG.md`)

### Phase 2 — Data Engineering & Analytics Pipeline
- Configured YAML-driven pipeline parameters (`configs/config.yaml`)
- Implemented modular preprocessing pipeline (`src/preprocessing/`):
  - Schema-validated data loader (`data_loader.py`)
  - Iterative brand filter (`brand_filter.py`)
  - BFS thread reconstruction engine (`thread_builder.py`)
  - Rule-based text cleaner preserving emojis & tone (`text_cleaner.py`)
  - Multi-stage quality filter (`quality_filter.py`)
  - Deterministic CSV/JSON exporter (`exporter.py`)
- Created EDA analysis suite & visualizations (`src/analysis/` & `results/eda/`)
- Developed standalone Jupyter EDA notebook (`notebooks/01_exploratory_data_analysis.ipynb`)
- Generated automated Markdown analytics report (`results/eda/EDA_REPORT.md`)

### Phase 3 — Intent Taxonomy & Golden Dataset System
- Designed 12-intent e-commerce taxonomy across 6 categories (`planning/03_INTENT_TAXONOMY.md`)
- Implemented terminal-based interactive annotation tool (`src/annotation/annotation_tool.py`)
- Built canonical annotation schema & enum validators (`src/annotation/schema.py`)
- Created automated dataset validator (`src/annotation/validator.py`)
- Built deterministic golden dataset exporter (`src/annotation/exporter.py`)
- Documented golden evaluation dataset procedures (`data/golden/README_GOLDEN.md`)
- Comprehensive test suite covering preprocessing and annotation (`tests/`)

---

## Repository Structure Overview

```
hiver-support-ai/
├── README.md
├── REPORT.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── planning/
│   ├── 00_ARCHITECTURE.md
│   ├── 01_BRAND_SELECTION.md
│   ├── 02_LABEL_GUIDELINES.md
│   ├── 03_INTENT_TAXONOMY.md
│   └── DECISION_LOG.md
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── README_DATA.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── golden/
│       └── README_GOLDEN.md
│
├── src/
│   ├── preprocessing/
│   ├── analysis/
│   ├── annotation/
│   └── utils/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── results/
│   └── eda/
│       └── EDA_REPORT.md
│
└── tests/
    ├── test_preprocessing.py
    └── test_annotation.py
```

---

## Next Steps (Phase 4)

- **Intent Classifier**: Train baseline & Transformer-based intent models on processed data
- **Historical Reply Retrieval**: Construct vector index for similar past support resolutions
- **Escalation Engine**: Build rule-based & ML guardrails for human transfer decisions
- **Evaluation Harness**: Benchmark Phase 4 models against `data/golden/golden_dataset.csv`
