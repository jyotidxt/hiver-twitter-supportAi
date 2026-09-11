# Hiver Support AI

An AI-powered customer support analysis and automation engine built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset, tailored for **@AmazonHelp**.

## Project Overview

This project implements an end-to-end conversational customer support pipeline:

1. **Phase 1 — Architecture & Planning**: System design, brand selection, guidelines, decision log.
2. **Phase 2 — Data Engineering & EDA**: Thread reconstruction, text cleaning, quality filtering, and EDA analytics report.
3. **Phase 3 — Intent Taxonomy & Golden Dataset**: 12-intent e-commerce taxonomy, CLI annotation tool, and validation system.
4. **Phase 4 — AI Support Agent Engine**: Runnable inference pipeline combining Intent Classification, Semantic Evidence Retrieval, Grounded Reply Generation, and Rule-based Escalation.

---

## Running the AI Agent (Under 5 Minutes)

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/jyotidxt/hiver-twitter-supportAi.git
cd hiver-twitter-supportAi

# Install dependencies
pip install -r requirements.txt

# (Optional) Set LLM API Key for OpenAI or Gemini (template fallback used if omitted)
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run the AI Support Agent

#### Interactive Terminal Demo (Recommended)
```bash
python -m src.cli.run_agent
```
*Prompts for input and displays pretty-printed inference results with top intents, retrieved historical evidence, grounded replies, and escalation decisions.*

#### Single Direct Message Mode
```bash
python -m src.cli.run_agent --input "Where is my package? Order #12345 has not arrived."
```

#### Batch JSON Mode (Demo Dataset)
```bash
python -m src.cli.run_agent --file examples/sample_messages.json --json
```

---

### Expected Output Example

```json
{
  "customer_message": "Where is my package? Order #12345 has not arrived.",
  "predicted_intent": "order_status",
  "confidence": 0.85,
  "top_3_candidates": [
    ["order_status", 0.85],
    ["order_missing_wrong", 0.08],
    ["refund_request", 0.04]
  ],
  "generated_reply": "We'd be glad to look into your order status! Please send us a DM with your order number and email address so we can check tracking for you.",
  "escalation_decision": "AUTO_HANDLE",
  "escalation_reason": "High confidence intent 'order_status' eligible for automated response",
  "processing_time_ms": 12.4,
  "retrieved_examples": [
    {
      "conversation_id": "conv_hist_001",
      "customer_text": "Where is my order? Order #10293 has not arrived yet.",
      "brand_reply": "Please DM us your order number and full name so we can trace your shipment details.",
      "similarity_score": 0.5291
    }
  ]
}
```

---

## Run Data Pipeline & Tests

```bash
# Run preprocessing pipeline
python -m src.preprocessing.run_pipeline

# Run EDA report generator
python -m src.analysis.run_eda

# Run complete pytest suite (70 tests)
pytest tests/ -v
```

---

## Project Structure

```
hiver-support-ai/
├── README.md
├── REPORT.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── planning/                         # Architecture & taxonomy design
│   ├── 00_ARCHITECTURE.md
│   ├── 01_BRAND_SELECTION.md
│   ├── 02_LABEL_GUIDELINES.md
│   ├── 03_INTENT_TAXONOMY.md
│   └── DECISION_LOG.md
│
├── configs/
│   └── config.yaml                   # Central pipeline & agent config
│
├── data/                             # Data assets & golden dataset
│   ├── README_DATA.md
│   ├── raw/ (twitter_support.csv)
│   ├── interim/ (reconstructed_threads.csv)
│   ├── processed/ (processed_conversations.csv)
│   └── golden/ (README_GOLDEN.md, golden_dataset.csv)
│
├── src/
│   ├── preprocessing/                # Data pipeline modules
│   ├── analysis/                     # EDA & analytics modules
│   ├── annotation/                   # Golden dataset annotation tool
│   ├── classifier/                   # Intent Classifier (TF-IDF + LogisticReg)
│   ├── retrieval/                    # Semantic Retriever (SentenceEmbeddings/Cosine)
│   ├── reply_generator/              # Grounded Reply Generator & Prompts
│   ├── escalation/                   # Escalation Policy Engine
│   ├── pipeline/                     # InferenceEngine & SupportAgent
│   ├── cli/                          # Runnable CLI Demo (run_agent.py)
│   └── utils/                        # Config, Logger & Helpers
│
├── examples/
│   └── sample_messages.json          # Demo customer queries dataset
│
├── logs/
│   └── agent.log                     # Audit log file
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── results/eda/
│   └── EDA_REPORT.md                 # Generated analytics report
│
└── tests/                            # 70 unit and pipeline tests
    ├── test_preprocessing.py
    ├── test_annotation.py
    ├── test_classifier.py
    ├── test_retrieval.py
    ├── test_escalation.py
    └── test_pipeline.py
```

---

## Phases Overview

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Architecture, Brand Selection, Labeling Guidelines |
| **Phase 2** | ✅ Complete | Preprocessing Pipeline, Thread Reconstruction, EDA Suite |
| **Phase 3** | ✅ Complete | 12-Intent Taxonomy, Terminal Annotation Tool, Golden Dataset |
| **Phase 4** | ✅ Complete | AI Agent Engine, Classifier, Retriever, Reply Gen, Escalation CLI |
| **Phase 5** | 🔲 Planned  | Evaluation Harness, Metrics, LLM-as-Judge, Human Agreement |

---

## License

This project is for academic/assignment purposes.
