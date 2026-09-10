# Data Directory — Hiver Support AI

## Overview

This directory contains all data assets for the Hiver Support AI project.
The preprocessing pipeline transforms raw Twitter customer support data
into clean, structured conversation threads ready for annotation and
model development.

## Directory Structure

```
data/
├── raw/                              # Raw, unprocessed data
│   └── twitter_support.csv           # Kaggle dataset (not tracked in git)
├── interim/                          # Intermediate pipeline outputs
│   └── reconstructed_threads.csv     # Threads before quality filtering
├── processed/                        # Final pipeline output
│   ├── processed_conversations.csv   # Full processed dataset
│   ├── annotation_ready.csv          # Simplified annotation format
│   └── preprocessing_summary.json    # Pipeline run statistics
└── README_DATA.md                    # This file
```

## Data Source

**Dataset**: Customer Support on Twitter
**Source**: [Kaggle - thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
**Format**: CSV
**Size**: ~1.6M tweets across multiple brands

### Raw Schema

| Column | Type | Description |
|--------|------|-------------|
| `tweet_id` | string | Unique anonymized tweet ID |
| `author_id` | string | Unique anonymized author ID |
| `inbound` | boolean | `True` for customer messages, `False` for brand |
| `created_at` | datetime | Tweet timestamp |
| `text` | string | Tweet content |
| `response_tweet_id` | string | Comma-separated IDs of response tweets |
| `in_response_to_tweet_id` | string | ID of the tweet being replied to |

## Pipeline Stages

### Stage 1: Data Loading
- Reads the raw CSV file
- Validates column schema
- Reports dataset statistics and missing values

### Stage 2: Brand Filtering
- Filters tweets for the configured brand (default: AmazonHelp)
- Retains all customer tweets in conversations with the brand
- Iteratively expands to capture full conversation chains

### Stage 3: Thread Reconstruction
- Traces `in_response_to_tweet_id` chains to find root tweets
- Rebuilds complete conversation threads via BFS traversal
- Assigns `conversation_id` and `role` (customer/brand)
- Sorts messages chronologically within each thread
- Saves intermediate output to `data/interim/reconstructed_threads.csv`

### Stage 4: Text Cleaning
- Removes URLs
- Removes RT prefixes
- Strips leading @mentions from replies
- Normalizes whitespace
- Preserves emojis and meaningful punctuation
- Optionally lowercases text (configurable)
- Preserves original text in `text_original` column

### Stage 5: Quality Filtering
- Removes empty/null text messages
- Removes system and deleted messages
- Removes messages shorter than minimum length
- Removes duplicate messages (same author + same text)
- Removes conversations shorter than minimum thread length
- All removal counts are tracked and reported

### Stage 6: Dataset Export
- `processed_conversations.csv`: Full processed dataset
- `annotation_ready.csv`: One row per conversation with formatted dialogue
- `preprocessing_summary.json`: Statistics and metadata

## Processed File Schemas

### processed_conversations.csv

| Column | Description |
|--------|-------------|
| `conversation_id` | Unique thread identifier (e.g., conv_000001) |
| `tweet_id` | Original tweet ID |
| `author_id` | Author identifier |
| `role` | `customer` or `brand` |
| `inbound` | Boolean inbound flag |
| `created_at` | Timestamp |
| `text` | Cleaned tweet text |
| `text_original` | Original text before cleaning |
| `in_response_to_tweet_id` | Reply chain reference |
| `response_tweet_id` | Response references |

### annotation_ready.csv

| Column | Description |
|--------|-------------|
| `conversation_id` | Unique thread identifier |
| `opening_message` | First customer message |
| `full_conversation` | Full dialogue with role labels |
| `num_messages` | Total messages in thread |
| `num_customer_messages` | Customer message count |
| `num_brand_replies` | Brand reply count |
| `customer_id` | Customer author ID |

## Reproducibility

### Prerequisites

1. Python 3.11+
2. Required packages: `pip install -r requirements.txt`
3. Raw dataset: Download from Kaggle and place at `data/raw/twitter_support.csv`

### Download the Dataset

```bash
# Kaggle CLI
kaggle datasets download -d thoughtvector/customer-support-on-twitter
unzip customer-support-on-twitter.zip -d data/raw/
mv data/raw/twcs.csv data/raw/twitter_support.csv
```

### Run the Pipeline

```bash
python -m src.preprocessing.run_pipeline
python -m src.analysis.run_eda
```

### Configuration

All parameters are in `configs/config.yaml`. Key settings:

- `brand.selected_brand`: Target brand
- `threads.minimum_thread_length`: Minimum messages per conversation
- `preprocessing.lowercase`: Whether to lowercase text
- `quality.min_text_length`: Minimum character threshold
