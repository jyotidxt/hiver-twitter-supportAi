# Hiver Support AI

An AI-powered customer support analysis pipeline built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

## Project Overview

This project builds a reproducible pipeline that:

1. **Phase 1** — Architecture & planning documents
2. **Phase 2** — Data engineering pipeline + exploratory data analysis
   - Loads raw Twitter support data
   - Filters for a single configurable brand
   - Reconstructs conversation threads
   - Cleans and normalizes text
   - Removes unusable records
   - Discovers candidate intent themes
   - Generates analytics report

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/jyotidxt/hiver-twitter-supportAi.git
cd hiver-twitter-supportAi

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter
# Extract to data/raw/twitter_support.csv
```

### Run Phase 2 — Data Pipeline

```bash
# Step 1: Run preprocessing pipeline
python -m src.preprocessing.run_pipeline

# Step 2: Run EDA & Brand Intelligence
python -m src.analysis.run_eda
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
hiver-support-ai/
├── README.md
├── .gitignore
├── requirements.txt
│
├── planning/                         # Phase 1 planning documents
│   ├── 00_ARCHITECTURE.md
│   ├── 01_BRAND_SELECTION.md
│   ├── 02_LABEL_GUIDELINES.md
│   └── DECISION_LOG.md
│
├── configs/
│   └── config.yaml                   # Pipeline configuration
│
├── data/
│   ├── README_DATA.md
│   ├── raw/                          # Raw Kaggle dataset
│   │   └── twitter_support.csv
│   ├── interim/                      # Intermediate outputs
│   │   └── reconstructed_threads.csv
│   └── processed/                    # Final pipeline outputs
│       ├── processed_conversations.csv
│       ├── annotation_ready.csv
│       └── preprocessing_summary.json
│
├── src/
│   ├── preprocessing/                # Data pipeline modules
│   │   ├── data_loader.py
│   │   ├── brand_filter.py
│   │   ├── thread_builder.py
│   │   ├── text_cleaner.py
│   │   ├── quality_filter.py
│   │   ├── exporter.py
│   │   └── run_pipeline.py
│   │
│   ├── analysis/                     # EDA & analytics modules
│   │   ├── dataset_overview.py
│   │   ├── conversation_analysis.py
│   │   ├── intent_discovery.py
│   │   ├── visualization.py
│   │   ├── report_generator.py
│   │   └── run_eda.py
│   │
│   └── utils/                        # Shared utilities
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── results/
│   └── eda/                          # Generated charts & reports
│       └── EDA_REPORT.md
│
└── tests/
    └── test_preprocessing.py
```

## Configuration

All pipeline settings are in `configs/config.yaml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `brand.selected_brand` | Target brand | `AmazonHelp` |
| `threads.minimum_thread_length` | Min messages per thread | `2` |
| `preprocessing.lowercase` | Lowercase text | `false` |
| `preprocessing.remove_urls` | Remove URLs | `true` |
| `quality.remove_duplicates` | Remove duplicate messages | `true` |
| `analysis.n_clusters` | Intent discovery clusters | `8` |

## Generated Outputs

### Data Pipeline
| File | Description |
|------|-------------|
| `data/processed/processed_conversations.csv` | Full processed dataset |
| `data/processed/annotation_ready.csv` | One row per conversation |
| `data/processed/preprocessing_summary.json` | Pipeline statistics |

### EDA & Analytics
| File | Description |
|------|-------------|
| `results/eda/EDA_REPORT.md` | Comprehensive analysis report |
| `results/eda/*.png` | Visualization charts |
| `results/eda/dataset_overview.json` | Machine-readable stats |

## Tech Stack

- **Python 3.11** — Core language
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Visualizations
- **scikit-learn** — TF-IDF + K-Means clustering
- **PyYAML** — Configuration
- **tqdm** — Progress bars
- **pytest** — Testing

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Architecture & planning |
| Phase 2 | ✅ Complete | Data engineering & EDA |
| Phase 3 | 🔲 Planned | Intent classification |
| Phase 4 | 🔲 Planned | Reply generation |

## License

This project is for academic/assignment purposes.
