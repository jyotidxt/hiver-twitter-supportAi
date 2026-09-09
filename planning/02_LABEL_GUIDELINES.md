# planning/02_LABEL_GUIDELINES.md

## 1. Purpose

- **Golden Evaluation Dataset** – a small, high‑quality, manually annotated set of 150‑250 customer messages (with context) that serves as the reference benchmark for all model‑level evaluations.
- **Why manual labels?**  Human judgements capture subtleties such as sarcasm, implicit intent, and policy nuances that automatic heuristics typically miss. Consequently, the gold set provides a trustworthy ground truth for measuring intent‑classification accuracy, reply‑generation fidelity, and escalation decisions.
- **Evaluation‑only use** – the dataset is **not** used for training or model fine‑tuning. It is held out for periodic validation, ablation studies, and for reporting reproducible benchmark numbers.

## 2. Annotation Unit

Each annotation row represents **one incoming customer tweet** (the *target message*) together with any preceding conversation context that is required to resolve intent or escalation.  The annotator is provided with:
- the raw customer text,
- the full thread of prior brand and customer messages (if any),
- metadata such as tweet timestamp.
The unit does **not** include follow‑up customer messages after the target tweet.

## 3. Intent Taxonomy

| Intent | Description | When to Use | When NOT to Use | Fictional Example |
|--------|-------------|-------------|-----------------|-------------------|
| **Account Inquiry** | Questions about account status, login, profile updates. | Customer asks "How do I reset my password?" | Not about billing or shipping. | *"I can't log into my account; can you help?"* |
| **Order Status** | Requests for the current state of a placed order. | Customer mentions order ID and asks for update. | No order reference present. | *"What's the ETA for order #12345?"* |
| **Billing Issue** | Disputes, invoices, over‑charges, refunds related to payment. | Explicit mention of charge, invoice, or refund. | General product question. | *"I was charged twice for my last purchase; can I get a refund?"* |
| **Technical Problem** | Issues with product functionality, app crashes, connectivity. | Describes error messages or malfunction. | Purely policy‑related queries. | *"The app crashes every time I open the chat feature."* |
| **Feature Request** | Suggestions for new features or improvements. | Proposes a new capability. | Complaints about existing issues. | *"It would be great if you added a dark mode."* |
| **Shipping / Delivery** | Questions about shipping method, address changes, delivery problems. | Mentions shipping, address, or delivery timeline. | No logistics component. | *"My package was delivered to the wrong address; what can be done?"* |
| **General Praise / Feedback** | Positive feedback, compliments, or satisfaction statements. | Expresses gratitude or praise without a problem. | Implicit complaint hidden in sarcasm (see edge cases). | *"Your support team was super helpful, thanks!"* |
| **Policy Clarification** | Requests for clarification of company policies, terms, or warranties. | Asks about return policy, warranty period, etc. | Specific technical or billing issue. | *"What is your 30‑day return policy?"* |
| **Escalation Request** | Customer explicitly asks for a human or higher‑level assistance. | Direct request such as "talk to a manager". | Implicit dissatisfaction without request. | *"I need to speak with a supervisor about this issue."* |
| **Other / Unclear** | Any message that does not fit the above categories or is ambiguous. | When intent cannot be confidently mapped. | When a clear intent matches another slot. | *"???"* |

*The taxonomy is intentionally concise (11 intents) to promote consistent labeling across annotators while covering the most common support scenarios for a single brand.*

## 4. Escalation Labels

| Label | Definition | Decision Criteria | Fictional Example |
|-------|------------|-------------------|-------------------|
| **Auto Handle** | The issue can be resolved automatically by the AI agent without human involvement. | High confidence intent, clear resolution path, no policy or security constraints. | *Customer asks for order status and the system can retrieve it instantly.* |
| **Human Required** | The conversation must be transferred to a human operator. | Contains sensitive data, policy exceptions, or the AI lacks sufficient confidence. | *Customer requests a refund for a damaged product.* |
| **Uncertain** | Annotator is unable to decide definitively whether automation is safe. | Ambiguous language, missing information, or mixed signals. | *Customer says "This is not what I expected" without specifying the problem.* |

## 5. Human Escalation Reasons

| Reason | Description |
|--------|------------|
| **Identity Verification** | Need to confirm the customer's identity before proceeding (e.g., password reset, account changes). |
| **Refund / Payment** | Request for monetary compensation, charge reversal, or billing correction. |
| **Legal / Privacy** | Queries involving legal obligations, data‑privacy requests, or compliance matters. |
| **Account Security** | Concerns about compromised accounts, unauthorized access, or security breaches. |
| **Damaged / Defective Product** | Report of a physical product that arrived broken or malfunctioning. |
| **Abusive Customer** | Customer uses profanity, harassment, or violates terms of service. |
| **Missing Information** | Essential details (order ID, account number) are absent, preventing resolution. |
| **Policy Exception** | Situation falls outside standard policy (e.g., special discount, VIP handling). |

## 6. Annotation Workflow

1. **Read Customer Tweet** – focus on the target message.
2. **Read Previous Context** – scroll up the thread to understand prior exchanges.
3. **Assign Intent** – select the most specific intent from the taxonomy.
4. **Decide Escalation** – choose *Auto Handle*, *Human Required*, or *Uncertain*.
5. **Record Reason** – if *Human Required* is chosen, pick the appropriate escalation reason from the controlled list.
6. **Add Notes** – optional free‑text field for ambiguities, justification, or special observations.
7. **Save Row** – ensure all required columns are filled before moving to the next tweet.

## 7. Edge Cases

- **Multiple Issues in One Tweet** – label the *primary* intent (the one that drives the next action). Add a note mentioning secondary issues.
- **Sarcasm** – rely on context and tone cues; if intent cannot be inferred, use *Uncertain* with a note.
- **Emojis** – treat emojis as sentiment cues but do not let them override textual intent.
- **Extremely Short Messages** – if the message is a single word like "Help", use *Uncertain* unless context clarifies.
- **Typo‑Heavy Text** – attempt to decode the meaning; if ambiguous, mark *Uncertain* and note the difficulty.
- **Customer Replies Only "Thanks"** – treat as *General Praise / Feedback* intent; escalation label is *Auto Handle*.
- **Duplicate Messages** – if identical tweets appear in the same thread, annotate only the first occurrence; subsequent duplicates are marked as *Duplicate* in notes and excluded from the final count.

## 8. Quality Control

- **Double Review** – each annotated row is independently reviewed by a second annotator.
- **Disagreement Resolution** – annotators discuss discrepancies; a senior lead makes the final decision.
- **Consistency Checks** – weekly audits of random samples to verify adherence to taxonomy definitions.
- **Label Revision Log** – maintain a `label_revision_log.csv` recording row ID, original label, revised label, reviewer, and timestamp.

## 9. Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique identifier for the target tweet (e.g., tweet ID). |
| `conversation_id` | string | Identifier for the entire thread; useful for grouping. |
| `customer_text` | string | Raw text of the incoming customer tweet. |
| `context` | string | Concatenated prior messages in the thread (chronological). |
| `intent` | string | One of the intent names from the taxonomy. |
| `escalation_label` | string | `Auto Handle` / `Human Required` / `Uncertain`. |
| `escalation_reason` | string (optional) | One of the controlled escalation reasons; required only when `Human Required` is selected. |
| `annotator_notes` | string (optional) | Free‑form comments about ambiguities, edge‑case handling, or rationale. |
| `annotator_id` | string | Identifier of the annotator who created the row. |

All columns are **required** except `escalation_reason` (when not applicable) and `annotator_notes`.

## 10. Acceptance Checklist

- [ ] Customer tweet is the *target* message (not a follow‑up).
- [ ] Context includes all preceding messages up to the most recent brand reply.
- [ ] Intent label matches the taxonomy definition *exactly*.
- [ ] Escalation label is chosen according to the decision criteria.
- [ ] If `Human Required`, an escalation reason is provided from the controlled list.
- [ ] Annotator notes are added for any uncertainty or special circumstances.
- [ ] Row passes double‑review without unresolved disagreement.
- [ ] Entry is logged in the `label_revision_log.csv` if any changes were made after review.

---
*Prepared by the senior machine‑learning engineer and NLP dataset lead.*
