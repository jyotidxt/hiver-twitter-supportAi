"""Configuration loader for the preprocessing pipeline.

Loads YAML configuration and provides typed access to pipeline settings.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Project root is 3 levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class PipelineConfig:
    """Typed configuration wrapper for the preprocessing pipeline.

    Attributes:
        raw: The raw dictionary loaded from YAML.
        selected_brand: The brand to filter for.
        brand_handle: The Twitter handle of the selected brand.
        raw_data_path: Path to the raw CSV file.
        processed_dir: Path to the processed output directory.
        processed_conversations_path: Path for processed conversations CSV.
        annotation_ready_path: Path for annotation-ready CSV.
        preprocessing_summary_path: Path for the preprocessing summary JSON.
        minimum_thread_length: Minimum number of messages in a valid thread.
        max_thread_depth: Safety limit for thread reconstruction depth.
        lowercase: Whether to lowercase text during cleaning.
        remove_urls: Whether to remove URLs.
        normalize_whitespace: Whether to normalize whitespace.
        preserve_emojis: Whether to preserve emojis.
        preserve_punctuation: Whether to preserve meaningful punctuation.
        remove_rt_prefix: Whether to remove RT prefixes.
        strip_mentions_from_start: Whether to strip leading @mentions.
        min_text_length: Minimum character length after cleaning.
        remove_duplicates: Whether to remove duplicate messages.
        remove_empty: Whether to remove empty tweets.
        remove_system_messages: Whether to remove system/deleted messages.
        log_level: Logging level string.
        show_progress_bars: Whether to show tqdm progress bars.
        log_file: Path to the log file.
    """

    raw: dict[str, Any] = field(repr=False)

    # Brand
    selected_brand: str = ""
    brand_handle: str = ""

    # Paths
    raw_data_path: Path = Path()
    processed_dir: Path = Path()
    interim_dir: Path = Path()
    processed_conversations_path: Path = Path()
    annotation_ready_path: Path = Path()
    preprocessing_summary_path: Path = Path()

    # Threads
    minimum_thread_length: int = 2
    max_thread_depth: int = 20

    # Preprocessing flags
    lowercase: bool = False
    remove_urls: bool = True
    normalize_whitespace: bool = True
    preserve_emojis: bool = True
    preserve_punctuation: bool = True
    remove_rt_prefix: bool = True
    strip_mentions_from_start: bool = True

    # Quality
    min_text_length: int = 3
    remove_duplicates: bool = True
    remove_empty: bool = True
    remove_system_messages: bool = True

    # Logging
    log_level: str = "INFO"
    show_progress_bars: bool = True
    log_file: Path = Path()

    # EDA Analysis
    eda_output_dir: Path = Path()
    top_n_keywords: int = 30
    top_n_phrases: int = 20
    n_clusters: int = 8
    min_cluster_size: int = 5
    sample_messages_per_cluster: int = 5
    escalation_keywords: dict[str, list[str]] = field(default_factory=dict)
    viz_dpi: int = 150
    viz_format: str = "png"
    viz_colors: dict[str, str] = field(default_factory=dict)


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses the default
            path at configs/pipeline_config.yaml relative to project root.

    Returns:
        A PipelineConfig instance with all settings populated.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        KeyError: If required configuration keys are missing.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    _validate_required_keys(raw)

    brand = raw.get("brand", {})
    paths = raw.get("paths", {})
    exports = paths.get("exports", {})
    threads = raw.get("threads", {})
    preprocessing = raw.get("preprocessing", {})
    quality = raw.get("quality", {})
    logging_cfg = raw.get("logging", {})
    analysis_cfg = raw.get("analysis", {})
    viz_cfg = analysis_cfg.get("visualization", {})

    return PipelineConfig(
        raw=raw,
        # Brand
        selected_brand=brand.get("selected_brand", "AmazonHelp"),
        brand_handle=brand.get("brand_handle", "@AmazonHelp"),
        # Paths (resolve relative to project root)
        raw_data_path=PROJECT_ROOT / paths.get("raw_data", "data/raw/twcs.csv"),
        processed_dir=PROJECT_ROOT / paths.get("processed_dir", "data/processed"),
        interim_dir=PROJECT_ROOT / paths.get("interim_dir", "data/interim"),
        processed_conversations_path=PROJECT_ROOT / exports.get(
            "processed_conversations", "data/processed/processed_conversations.csv"
        ),
        annotation_ready_path=PROJECT_ROOT / exports.get(
            "annotation_ready", "data/processed/annotation_ready.csv"
        ),
        preprocessing_summary_path=PROJECT_ROOT / exports.get(
            "preprocessing_summary", "results/preprocessing_summary.json"
        ),
        # Threads
        minimum_thread_length=threads.get("minimum_thread_length", 2),
        max_thread_depth=threads.get("max_thread_depth", 20),
        # Preprocessing
        lowercase=preprocessing.get("lowercase", False),
        remove_urls=preprocessing.get("remove_urls", True),
        normalize_whitespace=preprocessing.get("normalize_whitespace", True),
        preserve_emojis=preprocessing.get("preserve_emojis", True),
        preserve_punctuation=preprocessing.get("preserve_punctuation", True),
        remove_rt_prefix=preprocessing.get("remove_rt_prefix", True),
        strip_mentions_from_start=preprocessing.get("strip_mentions_from_start", True),
        # Quality
        min_text_length=quality.get("min_text_length", 3),
        remove_duplicates=quality.get("remove_duplicates", True),
        remove_empty=quality.get("remove_empty", True),
        remove_system_messages=quality.get("remove_system_messages", True),
        # Logging
        log_level=logging_cfg.get("level", "INFO"),
        show_progress_bars=logging_cfg.get("show_progress_bars", True),
        log_file=PROJECT_ROOT / logging_cfg.get("log_file", "results/pipeline.log"),
        # EDA Analysis
        eda_output_dir=PROJECT_ROOT / analysis_cfg.get("eda_output_dir", "results/eda"),
        top_n_keywords=analysis_cfg.get("top_n_keywords", 30),
        top_n_phrases=analysis_cfg.get("top_n_phrases", 20),
        n_clusters=analysis_cfg.get("n_clusters", 8),
        min_cluster_size=analysis_cfg.get("min_cluster_size", 5),
        sample_messages_per_cluster=analysis_cfg.get("sample_messages_per_cluster", 5),
        escalation_keywords={
            cat: kws
            for cat, kws in analysis_cfg.get("escalation_keywords", {}).items()
            if isinstance(kws, list)
        },
        viz_dpi=viz_cfg.get("figure_dpi", 150),
        viz_format=viz_cfg.get("figure_format", "png"),
        viz_colors=viz_cfg.get("color_palette", {}),
    )


def _validate_required_keys(raw: dict[str, Any]) -> None:
    """Validate that the raw config contains all required top-level keys.

    Args:
        raw: The raw configuration dictionary.

    Raises:
        KeyError: If a required key is missing.
    """
    required_sections = ["brand", "paths"]
    for section in required_sections:
        if section not in raw:
            raise KeyError(
                f"Missing required configuration section: '{section}'. "
                f"Please check your pipeline_config.yaml."
            )

    if "selected_brand" not in raw.get("brand", {}):
        raise KeyError(
            "Missing required key 'brand.selected_brand' in configuration."
        )
