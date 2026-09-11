# Golden Dataset — Hiver Support AI

## Purpose

The golden dataset is a curated set of **150–250 manually annotated conversations** that serves as the ground truth for evaluating all AI models in this pipeline.

> **Important**: The golden dataset is for **evaluation only**. It must never be used as training data.

## Files

| File | Description |
|------|-------------|
| `annotation_progress.csv` | Raw annotation progress (append-only, supports resume) |
| `golden_dataset.csv` | Final validated and deduplicated export |
| `validation_report.txt` | Quality validation results |
| `README_GOLDEN.md` | This documentation |

## Annotation Workflow

### 1. Start Annotating

```bash
python -m src.annotation.annotation_tool
```

Optional flags:
- `--limit 50` — Annotate at most 50 conversations
- `--no-resume` — Start fresh (ignore previous progress)

### 2. For Each Conversation

The tool displays:
- Conversation ID
- Complete chronological thread with role labels
- Customer opening message
- Message counts

You provide:
1. **Primary Intent** — Select from the 12-intent taxonomy (see `planning/03_INTENT_TAXONOMY.md`)
2. **Escalation** — Does this require human intervention? (Yes/No)
3. **Escalation Reason** — If yes, select from controlled list
4. **Confidence** — Your confidence level (High/Medium/Low)
5. **Notes** — Optional free-text observations

### 3. Resume Interrupted Sessions

Progress is saved after every annotation. Simply re-run the tool — it automatically skips completed conversations.

### 4. Validate & Export

```bash
# Validate only
python -m src.annotation.exporter --validate-only

# Validate and export
python -m src.annotation.exporter
```

## Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `conversation_id` | string | Yes | Unique conversation identifier |
| `primary_intent` | string | Yes | One of 12 intent labels from taxonomy |
| `escalation` | boolean | Yes | Whether human escalation is needed |
| `escalation_reason` | string | Conditional | Required when escalation=True |
| `confidence` | string | Yes | `high`, `medium`, or `low` |
| `annotator_notes` | string | No | Free-text observations |
| `annotated_at` | string | Yes | ISO 8601 UTC timestamp |

## Valid Intent Labels

| # | Intent | Category |
|---|--------|----------|
| 1 | `order_status` | Orders |
| 2 | `order_modification` | Orders |
| 3 | `order_missing_wrong` | Orders |
| 4 | `refund_request` | Payments & Refunds |
| 5 | `billing_issue` | Payments & Refunds |
| 6 | `delivery_problem` | Shipping & Delivery |
| 7 | `shipping_inquiry` | Shipping & Delivery |
| 8 | `account_access` | Account & Access |
| 9 | `account_management` | Account & Access |
| 10 | `product_issue` | Product & Technical |
| 11 | `technical_support` | Product & Technical |
| 12 | `general_inquiry` | General |

## Valid Escalation Reasons

| Reason | When to Use |
|--------|-------------|
| `identity_verification` | Customer identity needs confirmation |
| `refund_payment` | Monetary compensation or charge reversal |
| `legal_privacy` | Legal or data privacy concerns |
| `account_security` | Compromised account or unauthorized access |
| `damaged_defective` | Physical product damage report |
| `abusive_customer` | Profanity, harassment, TOS violation |
| `missing_information` | Essential details missing for resolution |
| `policy_exception` | Outside standard policy |

## Validation Rules

Before export, the validator checks:

1. No duplicate conversation IDs
2. All intents are from the valid taxonomy
3. Confidence is one of: high, medium, low
4. Escalation reason is provided when escalation=True
5. No empty conversation IDs
6. No empty intent fields

## Reproducibility

- `annotation_progress.csv` is append-only — it preserves the full annotation history
- `golden_dataset.csv` is generated deterministically from progress (sorted by conversation_id, last annotation wins for duplicates)
- Both files are included in version control
