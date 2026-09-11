# Architecture & Engineering Decision Log

This document logs all major architectural, engineering, and methodological choices made during the development of the **Hiver Support AI** system.

---

## Decision Summary Table

| # | Topic | Decision | Why Chosen | Alternative Considered | Engineering Trade-off |
|---|-------|----------|------------|------------------------|-----------------------|
| **1** | Brand Selection | Focus exclusively on `@AmazonHelp` | Highest volume in Kaggle dataset; diverse customer issues (shipping, returns, billing, tech). | Multi-brand joint training or `@AppleSupport` | Higher single-brand domain fidelity vs. lack of cross-brand generalization. |
| **2** | Thread Reconstruction | BFS Tree Traversal algorithm | Reliably reconstructs parent-child tweet chains into clean customer-brand dialogue pairs. | Naive pairwise message matching | Handles multi-reply branching trees; requires recursive memory overhead. |
| **3** | Intent Taxonomy | 12-Intent E-Commerce Hierarchy | High coverage of e-commerce support workflows; balances granular routing with low annotation ambiguity. | Banking77 taxonomy (77 intents) or 3-intent generic classification | 12 intents provide actionable routing without overwhelming manual annotators. |
| **4** | Preprocessing Rules | Preserve emojis and punctuation | Customer sentiment and urgency signals (e.g., `‼️`, `???`) are essential for escalation rules. | Aggressive lowercasing & punctuation removal | Preserves sentiment signals vs. slightly larger vocabulary size. |
| **5** | Classifier Model | TF-IDF + Logistic Regression Baseline | Fast, lightweight, highly interpretable baseline with probability outputs; no GPU required. | Fine-tuned DistilBERT / DeBERTa transformer | Instant CPU training & 12ms inference latency vs. lower semantic contextual awareness. |
| **6** | Evidence Retrieval | Semantic Retriever + Sentence Embeddings | Anchors generated replies in verified historical resolutions, preventing hallucinated policies. | Direct LLM prompting without retrieval context | Eliminates policy hallucinated claims; bounded by historical evidence quality. |
| **7** | Grounded Reply Generator | Decoupled Prompt Templates (`prompts.py`) | Separates prompt engineering from core python logic; supports OpenAI, Gemini, & template fallback. | Hardcoded prompt strings inside generator logic | Clean separation of concerns & API independence vs. maintenance of prompt templates. |
| **8** | Escalation Architecture | Explicit Policy Rule Engine | Separates safety guardrails (confidence thresholding, keywords, sensitive intents) from classification. | Joint multi-task NN classification | 100% deterministic safety control over high-risk cases vs. manual rule tuning. |
| **9** | Unified API Entry Point | `analyze_customer_message()` | Single modular entry point for the entire repository returning structured JSON. | Fragmented multi-file script invocation | Simplified developer experience & CLI integration vs. encapsulation complexity. |
| **10** | Golden Dataset Design | Held-out annotation benchmark (150–250 samples) | Evaluation dataset remains strictly isolated from model training to ensure un-biased benchmarking. | K-fold cross-validation on training data | Guarantees true out-of-sample evaluation vs. smaller test set sample size. |
| **11** | Baseline Comparison | Two explicit baselines (Trivial & Simple) | Baseline 1 (Majority class) & Baseline 2 (TF-IDF Nearest Neighbor) establish zero/non-LLM skill bounds. | Single model evaluation without baselines | Proves true incremental value of Phase 4 AI Support Agent over simpler approaches. |
| **12** | Response Quality Metric | Qualitative LLM-as-a-Judge (1–5 Rubric) | Evaluates groundedness, correctness, empathy, actionability, and tone beyond n-gram overlap. | Overlap metrics (BLEU, ROUGE) | Measures actual semantic & policy quality vs. API cost for LLM judge execution. |
| **13** | Judge Validation | Human-LLM Agreement Analysis | Computes Cohen's Kappa, MAE, & exact match % between human annotators and LLM Judge. | Unvalidated LLM Judge outputs | Proves LLM Judge reliability & alignment with human QA standards. |
| **14** | Report Generation | Automated Output-Driven `REPORT.md` | `ReportBuilder` populates numbers and comparison tables dynamically from evaluation output files. | Manually typed static Markdown report | Guarantees 100% data reproducibility and eliminates human data entry errors. |

---

## Detailed Architectural Decision Records (ADRs)

### ADR-001: Selection of @AmazonHelp as Target Brand
- **Context**: The Kaggle Twitter dataset contains ~2.8 million tweets across dozens of corporate brands.
- **Decision**: Filter exclusively for `@AmazonHelp`.
- **Alternatives Considered**: Training a multi-brand model or selecting `@AppleSupport`.
- **Rationale**: AmazonHelp represents the highest conversation volume in the dataset (~170k tweets) and covers a complete spectrum of customer support workflows (orders, returns, refunds, billing, prime accounts, and technical issues).
- **Trade-off**: The resulting pipeline is deeply optimized for e-commerce support patterns but would require retraining for non-retail domains.

### ADR-005: Baseline Classifier Selection (TF-IDF + Logistic Regression)
- **Context**: Phase 4 requires an intent classifier predicting 12 intent categories.
- **Decision**: Use `TfidfVectorizer` ($1,2$-grams) combined with `LogisticRegression` ($C=1.0$).
- **Alternatives Considered**: Fine-tuning a BERT/DeBERTa model.
- **Rationale**: Logistic Regression trains in under 2 seconds on CPU, yields calibrated probability distributions for confidence thresholding, and provides a clear, interpretable baseline.
- **Trade-off**: Lower handling of complex syntactic paraphrasing compared to deep transformer models, though mitigated by semantic retrieval.

### ADR-008: Policy Escalation Engine Design
- **Context**: Support systems must decide when an automated reply is safe vs. when human intervention is mandatory.
- **Decision**: Implement a dedicated, configurable `EscalationEngine` that evaluates classifier confidence thresholds ($\ge 0.65$), sensitive intent lists (`account_access`, `billing_issue`), and security/legal trigger keywords.
- **Alternatives Considered**: Predicting escalation via an ML classifier.
- **Rationale**: Human escalation policies must be 100% deterministic and auditable for security, legal, and compliance reasons. ML classifiers carry unpredictable edge-case failure modes.
- **Trade-off**: Requires manual keyword dictionary maintenance.

### ADR-012: Evaluation via LLM-as-a-Judge over BLEU/ROUGE
- **Context**: Text overlap metrics like BLEU penalize semantically correct responses that use different phrasing.
- **Decision**: Build a 5-dimension rubric (Groundedness, Correctness, Empathy, Actionability, Brand Tone) evaluated by an LLM Judge (`src/judge/llm_judge.py`).
- **Alternatives Considered**: Computing BLEU or ROUGE-L scores against reference tweets.
- **Rationale**: Customer support quality depends on policy adherence, politeness, and clear DM actionability—none of which are accurately captured by n-gram overlap.
- **Trade-off**: Requires LLM API calls, addressed by providing deterministic heuristic fallback evaluation.
