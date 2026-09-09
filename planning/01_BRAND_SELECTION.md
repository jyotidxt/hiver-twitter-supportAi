# planning/01_BRAND_SELECTION.md

## 1. Purpose

Selecting a single brand has a direct impact on every downstream component of the AI customer‑support pipeline:

- **Intent Classification** – the distribution of intents (e.g., complaint, question, praise) varies by brand; a well‑chosen brand provides a balanced set that enables a robust classifier.
- **Historical Resolution Retrieval** – the relevance of past tickets hinges on the consistency and volume of the brand’s previous replies. A brand with rich, repeatable resolutions makes the retrieval module more effective.
- **Reply Generation** – style, tone, and language conventions are brand‑specific. Picking a brand with a clear, stable voice simplifies fine‑tuning and reduces the risk of off‑brand generations.
- **Escalation Evaluation** – escalation patterns (e.g., frequency of complex issues) differ across industries. A brand that exhibits a realistic mix of escalations allows us to train and validate the escalation engine.

Hence, the brand‑selection step ensures that the data available for training, validation, and testing is representative, high‑quality, and aligned with the project’s technical goals.

## 2. Brand Selection Criteria

| Criterion | Description | Weight |
|-----------|-------------|--------|
| **Conversation Volume** | Total number of tweets (customer + brand) in the dataset for the brand. Higher volume provides more training material. | % |
| **Complete Thread Count** | Number of full customer ↔ brand conversation threads (both sides present). Enables multi‑turn modeling. | % |
| **Issue Diversity** | Breadth of distinct intent categories (e.g., shipping, billing, technical). A diverse set encourages a generalisable intent classifier. | % |
| **Reply Consistency** | How uniform the brand’s language, sign‑offs, and style are across replies. Consistency aids style‑transfer during generation. | % |
| **Multi‑turn Quality** | Proportion of threads with ≥ 2 brand replies (allowing context propagation). | % |
| **Noise Level** | Fraction of spam, deleted tweets, or duplicated content. Lower noise reduces cleaning effort. | % |
| **Suitability for Retrieval** | Presence of resolved tickets that can be matched to new queries (e.g., clear problem‑solution pairs). | % |
| **Annotation Ease** | Expected effort to manually label a validation set (e.g., clarity of tweets, limited jargon). | % |
| **Industry Relevance** | Alignment with realistic customer‑support scenarios (e.g., high‑touch services). | % |
| **Regulatory Simplicity** | Minimal constraints regarding privacy or proprietary information in the public tweets. | % |

*The weights must sum to 100 %; they will be determined after an initial exploratory analysis.*

## 3. Candidate Brands

| Brand (Handle) | Industry | Expected Conversation Style | Strengths | Weaknesses |
|----------------|----------|----------------------------|-----------|------------|
| `@DeltaAirLines` | Airline | Formal, policy‑driven, frequent use of flight numbers & dates. | High multi‑turn threads (flight issues, re‑bookings). Rich resolution examples. | Seasonal spikes; many retweets that add noise.
| `@Verizon` | Telecom | Technical, jargon‑heavy, mixture of troubleshooting and billing. | Large volume, diverse issue set (network, device, billing). | Frequent promotional tweets that may dilute support content.
| `@AmazonHelp` | E‑commerce | Concise, order‑centric, often includes order IDs. | Very high conversation volume, many complete threads. | Spam and promotional content; some automated bot replies.
| `@UberSupport` | Ride‑hailing / Food delivery | Conversational, includes location & timing details. | Good mix of real‑time complaints and rapid brand responses. | High rate of duplicate or templated replies.
| `@Starbucks` | Food & beverage | Friendly, brand‑voice focused, often includes emojis. | Consistent tone, clear resolution steps for refunds/loyalty.
| `@WalmartHelp` | Retail | Direct, issue‑oriented, often mentions store locations. | Strong multi‑turn support for inventory and returns. | Mixed with marketing messages; occasional duplicate threads.
| `@SpotifyCares` | Music streaming | Casual, quick acknowledgments, sometimes redirects to help articles. | Low noise, clear escalation flags. | Fewer deep‑troubleshooting threads; limited issue diversity.
| `@AirbnbHelp` | Travel accommodation | Narrative, includes booking IDs, longer threads. | Rich context for multi‑turn resolution. | Higher proportion of deleted tweets due to privacy concerns.

*These brands are known to be present in the Kaggle “Customer Support on Twitter” dataset and cover a range of industries and communication styles.*

## 4. Sampling Strategy

1. **Initial Random Sample** – draw a stratified random subset (e.g., 5 % of the full dataset) to estimate the distribution of brands without processing the entire dump.
2. **Thread Reconstruction** – use the `in_reply_to_status_id` field to rebuild full conversation threads for each candidate brand within the sample.
3. **Orphan Removal** – discard tweets that lack a corresponding brand reply or customer follow‑up (i.e., incomplete threads).
4. **Brand‑Reply Filtering** – keep only those replies whose `author_id` matches the candidate brand’s handle; exclude retweets or third‑party mentions.
5. **Spam & Duplicate Elimination** – apply heuristic filters (e.g., repetitive content, presence of typical spam keywords, identical timestamps) to remove noisy entries.
6. **Language Consistency** – retain only English‑language tweets (using language metadata or a fast language detector).

The resulting filtered sample will be used to compute the quantitative metrics required for the evaluation criteria.

## 5. Thread Quality Rules

| Rule | Requirement |
|------|--------------|
| **Minimum Customer Message** | At least one original customer tweet containing a clear problem or question. |
| **Required Brand Response** | At least one reply from the brand within 48 hours of the customer tweet. |
| **Multi‑turn Preference** | Prefer threads with ≥ 2 brand replies (enables context propagation). |
| **Language Consistency** | All messages in the thread must be English; mixed‑language threads are excluded. |
| **Handling Missing Tweets** | If a tweet in the chain is missing (e.g., deleted), the thread is discarded unless the missing part is the brand’s reply and a replacement cannot be inferred. |
| **Spam/Noise Threshold** | No more than 10 % of messages in a thread may be flagged as spam by the cleaning heuristics. |

Only threads satisfying all applicable rules are counted toward the “Complete Thread Count” metric.

## 6. Final Decision Method

A repeatable, data‑driven workflow:

1. **Extract Candidate Brands** – enumerate all distinct brand handles present in the full dataset.
2. **Compute Statistics** – for each brand, calculate the metrics listed in the selection criteria (volume, complete threads, diversity, etc.) on the filtered sample.
3. **Manual Spot‑Check** – randomly select ~50 threads per brand and visually assess style consistency, noise, and annotation ease.
4. **Score Each Criterion** – translate raw metric values into a normalized 0‑1 score (e.g., min‑max scaling) for each brand.
5. **Apply Weighted Sum** – multiply each normalized score by the pre‑determined weight and sum to obtain an overall brand score.
6. **Select Highest‑Scoring Brand** – choose the brand with the maximum aggregate score.
7. **Document Justification** – produce a short report summarising the scores, the manual observations, and the rationale for the final selection.

All steps are scripted (Python notebooks or shell pipelines) and logged, ensuring that the process can be reproduced without subjective bias.

## 7. Expected Output

After completing the brand‑selection phase, the following artifacts will be generated:

- **`selected_brand.txt`** – plain‑text file containing the chosen brand handle.
- **`brand_comparison_table.md`** – markdown table summarising each candidate’s raw metrics and weighted scores.
- **`thread_statistics.json`** – JSON file with per‑brand counts of total tweets, complete threads, multi‑turn ratios, and noise percentages.
- **`brand_selection_report.md`** – narrative justification linking the quantitative scores to the qualitative observations from the manual spot‑checks.
- **`selection_log.txt`** – chronological log of the executed commands, timestamps, and any decisions made during the process.

These artifacts will be stored under the `planning/` directory and will form the baseline for all subsequent phases of the project.

---
*Prepared by the senior machine‑learning engineer and data scientist.*
