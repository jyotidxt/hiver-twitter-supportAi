# AI Support Agent Engine — Pipeline Documentation

**Phase 4 — AI Support Agent Core**

## Overview

The **AI Support Agent Engine** provides an integrated, modular pipeline for automated customer support handling on Twitter data (specifically optimized for **@AmazonHelp**).

It executes four core tasks for every incoming customer query:
1. **Intent Classification**: Predicts the primary customer intent using TF-IDF + Logistic Regression.
2. **Semantic Evidence Retrieval**: Finds top-k similar historical resolved customer-brand conversation pairs using sentence embeddings.
3. **Grounded Reply Generation**: Synthesizes empathetic, concise, and policy-compliant support replies based on prompt templates and evidence.
4. **Escalation Decision Engine**: Applies configurable policies to decide whether a query can be `AUTO_HANDLE` or must `ESCALATE` to human agents.

---

## Architecture & Execution Flow

```
Incoming Customer Message
           │
           ▼
┌──────────────────────────┐
│    SupportAgent          │  (Unified Entry Point)
└──────────┬───────────────┘
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
┌──────────────────────────┐             ┌──────────────────────────┐
│    IntentClassifier      │             │    SemanticRetriever     │
│ (TF-IDF + LogisticReg)   │             │ (SentenceTransformers/   │
└──────────┬───────────────┘             │  Cosine Similarity)      │
           │ Intent + Confidence         └──────────┬───────────────┘
           │                                        │ Evidence Items
           ├────────────────────────────────────────┘
           │
           ├───► ┌──────────────────────────┐
           │     │    EscalationEngine      │ ──► EscalationDecision
           │     │ (Confidence/Policy Rules)│     (AUTO_HANDLE / ESCALATE)
           │     └──────────────────────────┘
           │
           └───► ┌──────────────────────────┐
                 │     ReplyGenerator       │ ──► Generated Reply
                 │ (OpenAI/Gemini/Template) │
                 └──────────────────────────┘
```

---

## Module Responsibilities

| Module | Location | Primary Responsibility | Key Output |
|--------|----------|------------------------|------------|
| **Intent Classifier** | `src/classifier/intent_classifier.py` | Predicts primary intent and confidence probabilities | `IntentPrediction` (intent, confidence, top 3 candidates) |
| **Semantic Retriever** | `src/retrieval/retriever.py` | Finds top-k historical resolved conversations as evidence | `RetrievedEvidence` (historical customer/brand pairs) |
| **Reply Generator** | `src/reply_generator/generator.py` & `prompts.py` | Grounded reply synthesis using LLMs or templates | String reply (concise, empathetic, policy-safe) |
| **Escalation Engine** | `src/escalation/escalation.py` | Rule-based policy checks for automated vs human transfer | `EscalationDecision` (`AUTO_HANDLE`/`ESCALATE` + reason) |
| **Support Agent** | `src/pipeline/support_agent.py` | Orchestrates all modules into single API interface | JSON-serializable dictionary |

---

## Single Entry Point Usage

```python
from src.pipeline.support_agent import analyze_customer_message

# Process any customer tweet/message
result = analyze_customer_message("Where is my package? Order #12345 hasn't arrived.")

print(result["predicted_intent"])     # "order_status"
print(result["confidence"])           # 0.85
print(result["escalation_decision"])  # "AUTO_HANDLE"
print(result["generated_reply"])      # "We'd be glad to look into your order status..."
```

---

## Configuration (`configs/config.yaml`)

```yaml
agent:
  classifier:
    model_type: "tfidf_logistic_regression"
    golden_dataset_path: "data/golden/golden_dataset.csv"
    ngram_range: [1, 2]
    max_features: 5000

  retrieval:
    embedding_model: "all-MiniLM-L6-v2"
    top_k: 3
    index_source: "data/processed/annotation_ready.csv"

  reply_generator:
    llm_provider: "template"  # Options: openai, gemini, template
    model_name: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"

  escalation:
    confidence_threshold: 0.65
    auto_escalate_intents:
      - "account_access"
      - "billing_issue"
```

---

## Extension Points

1. **Classifier Replacement**: Implement a custom classifier class adhering to `.train(df)` and `.predict(text)` interface (e.g., DistilBERT or DeBERTa) without altering downstream modules.
2. **Custom LLM Provider**: Extend `ReplyGenerator` in `src/reply_generator/generator.py` to support Anthropic or local Ollama models.
3. **Prompt Customization**: Modify system rules and user templates inside `src/reply_generator/prompts.py`.
4. **Policy Rules**: Add new keyword trigger categories or escalation conditions in `src/escalation/escalation.py`.
