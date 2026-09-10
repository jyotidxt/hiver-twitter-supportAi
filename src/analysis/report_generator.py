"""EDA report generator module.

Produces a comprehensive Markdown report combining all analysis results,
charts, and observations into a single document.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger, log_section

logger = get_logger(__name__)


def generate_report(
    overview_stats: dict[str, Any],
    conversation_stats: dict[str, Any],
    language_stats: dict[str, Any],
    intent_stats: dict[str, Any],
    escalation_stats: dict[str, Any],
    chart_paths: list[Path],
    config: PipelineConfig,
) -> Path:
    """Generate the comprehensive EDA report as Markdown.

    Args:
        overview_stats: Dataset overview statistics.
        conversation_stats: Conversation structure analysis.
        language_stats: Customer language analysis.
        intent_stats: Intent discovery results.
        escalation_stats: Escalation pattern observations.
        chart_paths: Paths to generated chart images.
        config: Pipeline configuration.

    Returns:
        Path to the generated report file.
    """
    log_section(logger, "GENERATING EDA REPORT")

    output_dir = config.eda_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "EDA_REPORT.md"

    brand = config.selected_brand
    bc = overview_stats.get("basic_counts", {})
    tl = overview_stats.get("thread_length", {})
    rb = overview_stats.get("role_balance", {})
    turn = conversation_stats.get("turn_structure", {})
    depth = conversation_stats.get("response_depth", {})
    qc = language_stats.get("question_vs_complaint", {})
    emoji = language_stats.get("emoji_usage", {})
    emo_punct = language_stats.get("emotional_punctuation", {})
    esc_overall = escalation_stats.get("overall_estimate", {})

    # Build chart reference helper
    chart_map: dict[str, str] = {}
    for cp in chart_paths:
        chart_map[cp.stem] = cp.name

    sections: list[str] = []

    # ── Title ──
    sections.append(f"# EDA Report — {brand}")
    sections.append("")
    sections.append(f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    sections.append(f"**Brand**: {brand}")
    sections.append(f"**Phase**: 2.2 — Exploratory Data Analysis & Brand Intelligence")
    sections.append("")
    sections.append("---")
    sections.append("")

    # ── Executive Summary ──
    sections.append("## 1. Executive Summary")
    sections.append("")
    sections.append(
        f"This report presents the exploratory data analysis of **{bc.get('total_conversations', 0):,}** "
        f"customer support conversations for **{brand}**, comprising "
        f"**{bc.get('total_messages', 0):,}** total messages. "
        f"The dataset includes **{bc.get('customer_messages', 0):,}** customer messages and "
        f"**{bc.get('brand_replies', 0):,}** brand replies from "
        f"**{bc.get('unique_customers', 0):,}** unique customers."
    )
    sections.append("")
    sections.append(
        f"Conversations average **{tl.get('mean', 0)}** messages per thread "
        f"(median: {tl.get('median', 0)}), with the longest thread containing "
        f"{tl.get('max', 0)} messages. "
        f"**{turn.get('multi_turn_pct', 0)}%** of conversations are multi-turn, "
        f"indicating a significant proportion of complex support interactions."
    )
    sections.append("")
    sections.append(
        f"An estimated **{esc_overall.get('pct', 0)}%** of conversations exhibit "
        f"at least one escalation signal, suggesting meaningful opportunities "
        f"for automated triage and escalation detection."
    )
    sections.append("")
    sections.append("---")
    sections.append("")

    # ── Dataset Insights ──
    sections.append("## 2. Dataset Insights")
    sections.append("")
    sections.append("### Summary Statistics")
    sections.append("")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Total Conversations | {bc.get('total_conversations', 0):,} |")
    sections.append(f"| Total Messages | {bc.get('total_messages', 0):,} |")
    sections.append(f"| Customer Messages | {bc.get('customer_messages', 0):,} |")
    sections.append(f"| Brand Replies | {bc.get('brand_replies', 0):,} |")
    sections.append(f"| Unique Customers | {bc.get('unique_customers', 0):,} |")
    sections.append(f"| Avg Thread Length | {tl.get('mean', 0)} |")
    sections.append(f"| Median Thread Length | {tl.get('median', 0)} |")
    sections.append(f"| Longest Conversation | {tl.get('max', 0)} messages |")
    sections.append(f"| Shortest Conversation | {tl.get('min', 0)} messages |")
    sections.append(f"| Customer:Brand Ratio | {rb.get('customer_to_brand_ratio', 0)}:1 |")
    sections.append("")

    # Embed chart
    if "03_customer_vs_brand" in chart_map:
        sections.append(f"![Customer vs Brand Messages]({chart_map['03_customer_vs_brand']})")
        sections.append("")

    sections.append("---")
    sections.append("")

    # ── Conversation Behavior ──
    sections.append("## 3. Conversation Behavior")
    sections.append("")
    sections.append("### Thread Structure")
    sections.append("")
    sections.append("| Category | Count | Percentage |")
    sections.append("|----------|-------|-----------|")
    sections.append(f"| Single-turn (2 messages) | {turn.get('single_turn_count', 0):,} | {turn.get('single_turn_pct', 0)}% |")
    sections.append(f"| Multi-turn (3+ messages) | {turn.get('multi_turn_count', 0):,} | {turn.get('multi_turn_pct', 0)}% |")
    sections.append("")

    sections.append("### Length Buckets")
    sections.append("")
    buckets = conversation_stats.get("length_buckets", {})
    if buckets:
        total_b = sum(buckets.values()) or 1
        sections.append("| Bucket | Conversations | Percentage |")
        sections.append("|--------|--------------|-----------|")
        for label, count in buckets.items():
            pct = round(count / total_b * 100, 1)
            sections.append(f"| {label} | {count:,} | {pct}% |")
        sections.append("")

    sections.append(f"### Response Depth")
    sections.append("")
    sections.append(f"- Mean exchange depth: **{depth.get('mean_depth', 0)}** rounds")
    sections.append(f"- Max exchange depth: **{depth.get('max_switches', 0)}** role switches")
    sections.append("")

    # Embed charts
    if "01_conversation_length_distribution" in chart_map:
        sections.append(f"![Conversation Length Distribution]({chart_map['01_conversation_length_distribution']})")
        sections.append("")
    if "06_multiturn_distribution" in chart_map:
        sections.append(f"![Multi-turn Distribution]({chart_map['06_multiturn_distribution']})")
        sections.append("")

    sections.append("---")
    sections.append("")

    # ── Customer Language ──
    sections.append("## 4. Customer Language Analysis")
    sections.append("")
    sections.append("### Message Types")
    sections.append("")
    sections.append("| Type | Count | Percentage |")
    sections.append("|------|-------|-----------|")
    sections.append(f"| Questions (?) | {qc.get('questions', 0):,} | {qc.get('questions_pct', 0)}% |")
    sections.append(f"| Complaint indicators | {qc.get('complaints', 0):,} | {qc.get('complaints_pct', 0)}% |")
    sections.append(f"| Exclamations (!) | {qc.get('exclamation_msgs', 0):,} | {qc.get('exclamation_pct', 0)}% |")
    sections.append("")

    sections.append("### Emotional Signals")
    sections.append("")
    sections.append(f"- Multiple !! messages: **{emo_punct.get('multiple_exclamations', 0):,}**")
    sections.append(f"- Multiple ?? messages: **{emo_punct.get('multiple_questions', 0):,}**")
    sections.append(f"- ALL CAPS messages: **{emo_punct.get('mostly_caps', 0):,}**")
    sections.append(f"- Messages with emojis: **{emoji.get('messages_with_emoji', 0):,}** ({emoji.get('emoji_pct', 0)}%)")
    sections.append("")

    sections.append("### Top Customer Keywords")
    sections.append("")
    top_kws = language_stats.get("top_keywords", [])[:15]
    if top_kws:
        sections.append("| Rank | Keyword | Frequency |")
        sections.append("|------|---------|-----------|")
        for i, kw in enumerate(top_kws, 1):
            sections.append(f"| {i} | {kw['word']} | {kw['count']:,} |")
        sections.append("")

    if "04_top_customer_keywords" in chart_map:
        sections.append(f"![Top Keywords]({chart_map['04_top_customer_keywords']})")
        sections.append("")

    sections.append("### Common Phrases")
    sections.append("")
    bigrams = language_stats.get("top_bigrams", [])[:10]
    if bigrams:
        sections.append("| Phrase | Frequency |")
        sections.append("|--------|-----------|")
        for bg in bigrams:
            sections.append(f"| {bg['phrase']} | {bg['count']:,} |")
        sections.append("")

    if "07_question_vs_complaint" in chart_map:
        sections.append(f"![Question vs Complaint]({chart_map['07_question_vs_complaint']})")
        sections.append("")

    sections.append("---")
    sections.append("")

    # ── Candidate Intents ──
    sections.append("## 5. Candidate Intents")
    sections.append("")
    sections.append("> **Note**: These are automatically discovered themes, NOT final intent labels.")
    sections.append("> Human review is required to define the final intent taxonomy.")
    sections.append("")

    clusters = intent_stats.get("clusters", [])
    if clusters:
        sections.append(f"The clustering algorithm identified **{len(clusters)}** candidate themes ")
        sections.append(f"from **{intent_stats.get('total_messages_clustered', 0):,}** customer opening messages.")
        sections.append("")

        for i, c in enumerate(clusters, 1):
            sections.append(f"### Theme {i}: {c['theme_label']}")
            sections.append("")
            sections.append(f"- **Size**: {c['size']:,} messages ({c['frequency_pct']}%)")
            top_words = ", ".join(kw["word"] for kw in c.get("top_keywords", [])[:8])
            sections.append(f"- **Keywords**: {top_words}")
            sections.append("")

            sections.append("**Representative messages:**")
            sections.append("")
            for msg in c.get("representative_messages", [])[:3]:
                sections.append(f"> “{msg['text']}”")
                sections.append("")

        if "05_candidate_intent_frequency" in chart_map:
            sections.append(f"![Candidate Intent Frequency]({chart_map['05_candidate_intent_frequency']})")
            sections.append("")
    else:
        sections.append("No clusters were generated. Review the intent discovery module output.")
        sections.append("")

    sections.append("---")
    sections.append("")

    # ── Escalation Observations ──
    sections.append("## 6. Escalation Observations")
    sections.append("")
    sections.append("> **Note**: These are observational signals, NOT model predictions.")
    sections.append("> They identify areas where automated escalation detection could add value.")
    sections.append("")

    sections.append("### Structural Signals")
    sections.append("")
    sections.append("| Signal | Count | % of Conversations |")
    sections.append("|--------|-------|-------------------|")
    rep = escalation_stats.get("repeated_followups", {})
    unr = escalation_stats.get("unresolved", {})
    sections.append(f"| Repeated follow-ups (3+ customer msgs) | {rep.get('count', 0):,} | {rep.get('pct', 0)}% |")
    sections.append(f"| Unresolved (customer last speaker) | {unr.get('count', 0):,} | {unr.get('pct', 0)}% |")
    sections.append("")

    sections.append("### Keyword-Based Signals")
    sections.append("")
    kw_signals = escalation_stats.get("keyword_signals", {})
    if kw_signals:
        sections.append("| Category | Count | % of Conversations |")
        sections.append("|----------|-------|-------------------|")
        for cat, data in kw_signals.items():
            sections.append(f"| {cat.title()} | {data.get('count', 0):,} | {data.get('pct', 0)}% |")
        sections.append("")

    sections.append(
        f"**Overall**: An estimated **{esc_overall.get('conversations_with_any_signal', 0):,}** "
        f"conversations ({esc_overall.get('pct', 0)}%) exhibit at least one escalation signal."
    )
    sections.append("")

    if "08_escalation_signals" in chart_map:
        sections.append(f"![Escalation Signals]({chart_map['08_escalation_signals']})")
        sections.append("")

    sections.append("---")
    sections.append("")

    # ── Key Takeaways ──
    sections.append("## 7. Key Takeaways")
    sections.append("")
    sections.append(f"1. **Dataset Scale**: {bc.get('total_conversations', 0):,} conversations provide "
                    f"sufficient volume for model development.")
    sections.append(f"2. **Thread Complexity**: {turn.get('multi_turn_pct', 0)}% of conversations are multi-turn, "
                    f"requiring context-aware modeling.")
    sections.append(f"3. **Customer Tone**: {qc.get('complaints_pct', 0)}% of messages contain complaint indicators, "
                    f"while {qc.get('questions_pct', 0)}% are questions.")
    sections.append(f"4. **Escalation Potential**: {esc_overall.get('pct', 0)}% of conversations show "
                    f"escalation signals that could benefit from automated triage.")
    sections.append(f"5. **Intent Diversity**: {len(clusters)} candidate themes were discovered, "
                    f"providing a starting point for intent taxonomy design.")
    sections.append("")
    sections.append("---")
    sections.append("")

    # ── Recommendations ──
    sections.append("## 8. Recommendations for Annotation")
    sections.append("")
    sections.append("Based on this analysis, the following recommendations are made for the annotation phase:")
    sections.append("")
    sections.append("1. **Start with the discovered themes** as a draft intent taxonomy. "
                    "Review representative messages from each cluster to validate and refine labels.")
    sections.append("2. **Prioritize multi-turn conversations** for annotation, as they represent "
                    "the most complex and valuable training examples.")
    sections.append("3. **Include escalation signals** as annotation dimensions — annotators should "
                    "flag conversations that require human intervention.")
    sections.append("4. **Use the annotation_ready.csv** file as the primary annotation input, "
                    "which provides pre-formatted conversation threads.")
    sections.append("5. **Sample across intent clusters** to ensure balanced representation "
                    "in the training set rather than random sampling.")
    sections.append("6. **Preserve customer tone** markers (emojis, punctuation patterns) "
                    "as they carry signal for sentiment and urgency detection.")
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append(f"*Report generated automatically by the Hiver Support AI EDA pipeline.*")

    # Write report
    report_content = "\n".join(sections)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"  Report saved: {report_path}")
    return report_path
