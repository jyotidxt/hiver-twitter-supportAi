"""Intent discovery and customer language analysis module.

Analyzes customer language patterns and discovers candidate intent themes
using TF-IDF vectorization and clustering. This is exploratory analysis
only — no permanent intent labels are assigned.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import PipelineConfig
from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

# Common English stop words (minimal set to avoid heavy dependencies)
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "get", "got", "us",
    "also", "would", "could", "one", "two", "like", "still", "even",
    "back", "much", "go", "going", "went", "make", "made", "know",
    "need", "want", "please", "thank", "thanks", "hi", "hello", "hey",
    "dm", "amp",
}


def analyze_customer_language(
    df: pd.DataFrame, config: PipelineConfig
) -> dict[str, Any]:
    """Analyze customer language patterns.

    Studies frequent keywords, complaint phrases, question vs complaint
    ratio, emotional punctuation, and emoji usage.

    Args:
        df: The processed conversations DataFrame.
        config: Pipeline configuration.

    Returns:
        Dictionary of language analysis results.
    """
    log_section(logger, "CUSTOMER LANGUAGE ANALYSIS")

    customer_df = df[df["role"] == "customer"].copy()
    customer_texts = customer_df["text"].fillna("").tolist()
    total_msgs = len(customer_texts)

    results: dict[str, Any] = {}

    # ── Top keywords ──
    word_counts = _count_words(customer_texts)
    top_keywords = word_counts.most_common(config.top_n_keywords)
    results["top_keywords"] = [
        {"word": word, "count": count} for word, count in top_keywords
    ]

    logger.info(f"  Top {config.top_n_keywords} customer keywords:")
    for word, count in top_keywords[:10]:
        bar = "\u2588" * min(int(count / top_keywords[0][1] * 30), 30)
        logger.info(f"    {word:<20} {count:>6,}  {bar}")
    if len(top_keywords) > 10:
        logger.info(f"    ... and {len(top_keywords) - 10} more")

    # ── Common phrases (bigrams + trigrams) ──
    bigrams = _count_ngrams(customer_texts, n=2)
    trigrams = _count_ngrams(customer_texts, n=3)
    top_phrases = bigrams.most_common(config.top_n_phrases)
    top_trigrams = trigrams.most_common(config.top_n_phrases)

    results["top_bigrams"] = [
        {"phrase": phrase, "count": count} for phrase, count in top_phrases
    ]
    results["top_trigrams"] = [
        {"phrase": phrase, "count": count} for phrase, count in top_trigrams
    ]

    logger.info("")
    logger.info(f"  Top complaint phrases (bigrams):")
    for phrase, count in top_phrases[:10]:
        logger.info(f"    \"{phrase}\"  \u2192  {count:,}")

    # ── Question vs Complaint ratio ──
    question_count = sum(1 for t in customer_texts if "?" in t)
    exclamation_count = sum(1 for t in customer_texts if "!" in t)
    # Complaints often use exclamation or negative words without question marks
    complaint_indicators = [
        "not", "never", "worst", "terrible", "horrible", "awful",
        "disappointed", "frustrated", "angry", "unacceptable",
        "problem", "issue", "wrong", "bad", "broken", "fail",
    ]
    complaint_count = sum(
        1 for t in customer_texts
        if any(w in t.lower() for w in complaint_indicators)
    )

    results["question_vs_complaint"] = {
        "questions": question_count,
        "questions_pct": round(question_count / total_msgs * 100, 1) if total_msgs > 0 else 0,
        "complaints": complaint_count,
        "complaints_pct": round(complaint_count / total_msgs * 100, 1) if total_msgs > 0 else 0,
        "exclamation_msgs": exclamation_count,
        "exclamation_pct": round(exclamation_count / total_msgs * 100, 1) if total_msgs > 0 else 0,
    }

    log_metric(logger, "Messages with ?", f"{question_count:,} ({results['question_vs_complaint']['questions_pct']}%)")
    log_metric(logger, "Messages with complaint words", f"{complaint_count:,} ({results['question_vs_complaint']['complaints_pct']}%)")
    log_metric(logger, "Messages with !", f"{exclamation_count:,} ({results['question_vs_complaint']['exclamation_pct']}%)")

    # ── Emotional punctuation ──
    multi_excl = sum(1 for t in customer_texts if re.search(r"!{2,}", t))
    multi_question = sum(1 for t in customer_texts if re.search(r"\?{2,}", t))
    caps_msgs = sum(
        1 for t in customer_texts
        if len(t) > 10 and sum(1 for c in t if c.isupper()) / len(t) > 0.5
    )

    results["emotional_punctuation"] = {
        "multiple_exclamations": multi_excl,
        "multiple_questions": multi_question,
        "mostly_caps": caps_msgs,
    }

    log_metric(logger, "Multiple !! messages", f"{multi_excl:,}")
    log_metric(logger, "Multiple ?? messages", f"{multi_question:,}")
    log_metric(logger, "CAPS-heavy messages", f"{caps_msgs:,}")

    # ── Emoji frequency ──
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U0000fe00-\U0000fe0f"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff]+",
        flags=re.UNICODE,
    )

    emoji_counter: Counter[str] = Counter()
    msgs_with_emoji = 0
    for t in customer_texts:
        emojis = emoji_pattern.findall(t)
        if emojis:
            msgs_with_emoji += 1
            for e in emojis:
                for char in e:
                    emoji_counter[char] += 1

    results["emoji_usage"] = {
        "messages_with_emoji": msgs_with_emoji,
        "emoji_pct": round(msgs_with_emoji / total_msgs * 100, 1) if total_msgs > 0 else 0,
        "top_emojis": [
            {"emoji": e, "count": c} for e, c in emoji_counter.most_common(10)
        ],
        "total_unique_emojis": len(emoji_counter),
    }

    log_metric(logger, "Messages with emojis", f"{msgs_with_emoji:,} ({results['emoji_usage']['emoji_pct']}%)")

    return results


def discover_intents(
    df: pd.DataFrame, config: PipelineConfig
) -> dict[str, Any]:
    """Discover candidate intent themes using TF-IDF and K-Means clustering.

    This is exploratory intent discovery, NOT classification.
    Results are candidate themes with representative messages
    to help humans decide final intent labels.

    Args:
        df: The processed conversations DataFrame.
        config: Pipeline configuration.

    Returns:
        Dictionary with discovered clusters, their keywords,
        representative messages, and frequency estimates.
    """
    log_section(logger, "INTENT DISCOVERY (EXPLORATORY)")
    logger.info("  Clustering customer messages to discover themes...")
    logger.info("  Note: These are candidate themes, NOT final labels.")

    # Use only customer opening messages (first customer message per conversation)
    customer_df = df[df["role"] == "customer"].copy()
    first_msgs = customer_df.groupby("conversation_id").first().reset_index()
    texts = first_msgs["text"].fillna("").tolist()
    conv_ids = first_msgs["conversation_id"].tolist()

    # Filter out very short texts
    valid_mask = [len(t.strip()) > 10 for t in texts]
    valid_texts = [t for t, v in zip(texts, valid_mask) if v]
    valid_conv_ids = [c for c, v in zip(conv_ids, valid_mask) if v]

    logger.info(f"  Valid opening messages for clustering: {len(valid_texts):,}")

    if len(valid_texts) < config.n_clusters * 2:
        logger.warning("  Not enough messages for meaningful clustering.")
        return {"error": "Insufficient data for clustering"}

    # TF-IDF vectorization
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        logger.warning("  scikit-learn not installed. Falling back to keyword-only analysis.")
        return _keyword_only_intent_discovery(valid_texts, config)

    logger.info("  Running TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=list(STOP_WORDS),
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
    except ValueError as e:
        logger.warning(f"  TF-IDF failed: {e}")
        return _keyword_only_intent_discovery(valid_texts, config)

    feature_names = vectorizer.get_feature_names_out()

    # K-Means clustering
    n_clusters = min(config.n_clusters, len(valid_texts) // 5)
    logger.info(f"  Running K-Means with {n_clusters} clusters...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(1000, len(valid_texts)),
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Extract cluster information
    clusters: list[dict[str, Any]] = []

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = int(cluster_mask.sum())

        if cluster_size < config.min_cluster_size:
            continue

        # Top keywords for this cluster (by centroid weights)
        centroid = kmeans.cluster_centers_[cluster_id]
        top_indices = centroid.argsort()[::-1][:10]
        top_words = [
            (str(feature_names[i]), round(float(centroid[i]), 4))
            for i in top_indices
            if centroid[i] > 0
        ]

        # Representative messages (closest to centroid)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_tfidf = tfidf_matrix[cluster_indices]
        centroid_reshaped = centroid.reshape(1, -1)
        # Compute distances
        distances = np.asarray(
            (cluster_tfidf - centroid_reshaped).power(2).sum(axis=1)
        ).flatten()
        closest_indices = distances.argsort()[:config.sample_messages_per_cluster]

        representative_msgs = [
            {
                "text": valid_texts[cluster_indices[i]][:200],
                "conversation_id": valid_conv_ids[cluster_indices[i]],
            }
            for i in closest_indices
        ]

        # Generate a candidate theme label from top keywords
        theme_label = " / ".join([w for w, _ in top_words[:3]])

        clusters.append({
            "cluster_id": cluster_id,
            "theme_label": theme_label,
            "size": cluster_size,
            "frequency_pct": round(cluster_size / len(valid_texts) * 100, 1),
            "top_keywords": [{"word": w, "weight": s} for w, s in top_words],
            "representative_messages": representative_msgs,
        })

    # Sort by size (most frequent first)
    clusters.sort(key=lambda c: c["size"], reverse=True)

    # Log results
    logger.info("")
    logger.info(f"  Discovered {len(clusters)} candidate themes:")
    logger.info("")
    for c in clusters:
        logger.info(
            f"  {Colors.BOLD}Cluster {c['cluster_id']}: {c['theme_label']}{Colors.RESET}"
        )
        logger.info(
            f"    Size: {c['size']:,} messages ({c['frequency_pct']}%)"
        )
        logger.info(f"    Top keywords: {', '.join(w['word'] for w in c['top_keywords'][:5])}")
        if c["representative_messages"]:
            logger.info(f"    Sample: \"{c['representative_messages'][0]['text'][:100]}...\"")
        logger.info("")

    results = {
        "n_clusters": n_clusters,
        "total_messages_clustered": len(valid_texts),
        "clusters": clusters,
    }

    return results


def analyze_escalation_patterns(
    df: pd.DataFrame, config: PipelineConfig
) -> dict[str, Any]:
    """Analyze escalation patterns in conversations.

    Identifies conversations with signals that may require human handling:
    - Repeated customer follow-ups
    - Unresolved conversations (customer is last speaker)
    - Payment/billing language
    - Account access issues
    - Damaged product terminology
    - High urgency language

    Args:
        df: The processed conversations DataFrame.
        config: Pipeline configuration.

    Returns:
        Dictionary with escalation pattern observations.
    """
    log_section(logger, "ESCALATION PATTERN EXPLORATION")
    logger.info("  Identifying potential escalation signals...")
    logger.info("  Note: These are observations only, NOT model predictions.")

    results: dict[str, Any] = {}
    total_convs = df["conversation_id"].nunique()

    # ── Repeated customer follow-ups ──
    # Conversations where customer sends 3+ messages
    role_counts = df.groupby(["conversation_id", "role"]).size().unstack(fill_value=0)
    if "customer" in role_counts.columns:
        repeat_followup = role_counts[role_counts["customer"] >= 3]
        results["repeated_followups"] = {
            "count": int(len(repeat_followup)),
            "pct": round(len(repeat_followup) / total_convs * 100, 1) if total_convs > 0 else 0,
            "description": "Conversations where customer sent 3+ messages (potential frustration)",
        }
        log_metric(logger, "Repeated followups (3+ msgs)", f"{len(repeat_followup):,}")

    # ── Unresolved conversations ──
    # Last message is from customer (brand didn't close the conversation)
    last_speakers = (
        df.sort_values("created_at" if "created_at" in df.columns else "tweet_id")
        .groupby("conversation_id")["role"]
        .last()
    )
    unresolved = int((last_speakers == "customer").sum())
    results["unresolved"] = {
        "count": unresolved,
        "pct": round(unresolved / total_convs * 100, 1) if total_convs > 0 else 0,
        "description": "Conversations ending with customer message (possibly unresolved)",
    }
    log_metric(logger, "Potentially unresolved", f"{unresolved:,}")

    # ── Keyword-based escalation signals ──
    escalation_kws = config.escalation_keywords
    customer_texts = df[df["role"] == "customer"][["conversation_id", "text"]].copy()
    customer_texts["text_lower"] = customer_texts["text"].fillna("").str.lower()

    # Aggregate customer text per conversation
    conv_texts = customer_texts.groupby("conversation_id")["text_lower"].apply(" ".join)

    category_results: dict[str, Any] = {}
    for category, keywords in escalation_kws.items():
        matched_convs = conv_texts[
            conv_texts.apply(lambda t: any(kw in t for kw in keywords))
        ]
        category_results[category] = {
            "count": int(len(matched_convs)),
            "pct": round(len(matched_convs) / total_convs * 100, 1) if total_convs > 0 else 0,
            "keywords_used": keywords,
            "sample_conversations": matched_convs.index.tolist()[:5],
        }
        log_metric(
            logger,
            f"{category.title()} language",
            f"{len(matched_convs):,} ({category_results[category]['pct']}%)",
        )

    results["keyword_signals"] = category_results

    # ── Overall escalation estimate ──
    # Combine signals: conversation has ANY escalation signal
    all_escalation_convs: set[str] = set()
    if "customer" in role_counts.columns:
        all_escalation_convs.update(repeat_followup.index.tolist())
    all_escalation_convs.update(
        last_speakers[last_speakers == "customer"].index.tolist()
    )
    for cat_data in category_results.values():
        all_escalation_convs.update(cat_data["sample_conversations"])
        # Add all matched conversations
        cat_kws = cat_data["keywords_used"]
        matched = conv_texts[
            conv_texts.apply(lambda t: any(kw in t for kw in cat_kws))
        ].index.tolist()
        all_escalation_convs.update(matched)

    results["overall_estimate"] = {
        "conversations_with_any_signal": len(all_escalation_convs),
        "pct": round(
            len(all_escalation_convs) / total_convs * 100, 1
        ) if total_convs > 0 else 0,
    }

    log_metric(
        logger,
        "Any escalation signal",
        f"{len(all_escalation_convs):,} ({results['overall_estimate']['pct']}%)",
    )

    return results


# ── Helper functions ──

def _count_words(texts: list[str]) -> Counter[str]:
    """Count word frequencies across texts, excluding stop words.

    Args:
        texts: List of text strings.

    Returns:
        Counter of word frequencies.
    """
    counter: Counter[str] = Counter()
    word_pattern = re.compile(r"\b[a-zA-Z]{2,}\b")

    for text in texts:
        words = word_pattern.findall(text.lower())
        counter.update(w for w in words if w not in STOP_WORDS)

    return counter


def _count_ngrams(texts: list[str], n: int = 2) -> Counter[str]:
    """Count n-gram frequencies across texts.

    Args:
        texts: List of text strings.
        n: N-gram size (2 for bigrams, 3 for trigrams).

    Returns:
        Counter of n-gram frequencies.
    """
    counter: Counter[str] = Counter()
    word_pattern = re.compile(r"\b[a-zA-Z]{2,}\b")

    for text in texts:
        words = [
            w for w in word_pattern.findall(text.lower())
            if w not in STOP_WORDS
        ]
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            counter[ngram] += 1

    return counter


def _keyword_only_intent_discovery(
    texts: list[str], config: PipelineConfig
) -> dict[str, Any]:
    """Fallback intent discovery using keyword frequency only.

    Used when scikit-learn is not available.

    Args:
        texts: List of customer message texts.
        config: Pipeline configuration.

    Returns:
        Dictionary with keyword-based theme discovery.
    """
    logger.info("  Using keyword-only intent discovery (no clustering).")

    word_counts = _count_words(texts)
    top_words = word_counts.most_common(50)

    # Group top words into rough themes
    themes = {
        "order_delivery": ["order", "delivery", "shipping", "package", "track", "delivered", "ship"],
        "refund_payment": ["refund", "charge", "payment", "money", "credit", "price", "billing"],
        "account_access": ["account", "login", "password", "email", "access", "verify"],
        "product_issue": ["broken", "damaged", "defective", "wrong", "missing", "item", "product"],
        "general_inquiry": ["help", "question", "information", "wondering", "available"],
    }

    theme_results = []
    for theme_name, theme_kws in themes.items():
        count = sum(word_counts.get(w, 0) for w in theme_kws)
        theme_results.append({
            "theme_label": theme_name.replace("_", " ").title(),
            "keywords": theme_kws,
            "total_mentions": count,
        })

    theme_results.sort(key=lambda x: x["total_mentions"], reverse=True)

    return {
        "method": "keyword_frequency",
        "note": "Fallback analysis without clustering (install scikit-learn for full analysis)",
        "themes": theme_results,
        "top_keywords": [{"word": w, "count": c} for w, c in top_words],
    }
