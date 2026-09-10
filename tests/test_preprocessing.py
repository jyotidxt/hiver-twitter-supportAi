"""Tests for the preprocessing pipeline modules.

Covers data loading, thread reconstruction, text cleaning,
quality filtering, and configuration loading.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════


@pytest.fixture
def mock_config() -> Any:
    """Create a mock PipelineConfig for testing."""
    from src.utils.config import PipelineConfig

    return PipelineConfig(
        raw={},
        selected_brand="TestBrand",
        brand_handle="@TestBrand",
        raw_data_path=Path("data/raw/test.csv"),
        processed_dir=Path("data/processed"),
        interim_dir=Path("data/interim"),
        processed_conversations_path=Path("data/processed/test_processed.csv"),
        annotation_ready_path=Path("data/processed/test_annotation.csv"),
        preprocessing_summary_path=Path("data/processed/test_summary.json"),
        minimum_thread_length=2,
        max_thread_depth=20,
        lowercase=False,
        remove_urls=True,
        normalize_whitespace=True,
        preserve_emojis=True,
        preserve_punctuation=True,
        remove_rt_prefix=True,
        strip_mentions_from_start=True,
        min_text_length=3,
        remove_duplicates=True,
        remove_empty=True,
        remove_system_messages=True,
        log_level="WARNING",
        show_progress_bars=False,
        log_file=Path("results/test.log"),
        eda_output_dir=Path("results/eda"),
        top_n_keywords=10,
        top_n_phrases=10,
        n_clusters=3,
        min_cluster_size=2,
        sample_messages_per_cluster=2,
        escalation_keywords={},
        viz_dpi=72,
        viz_format="png",
        viz_colors={},
    )


@pytest.fixture
def sample_tweets() -> pd.DataFrame:
    """Create a sample tweets DataFrame mimicking the raw dataset."""
    return pd.DataFrame({
        "tweet_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "author_id": [
            "customer1", "TestBrand", "customer1", "TestBrand",
            "customer2", "TestBrand", "OtherBrand", "customer3",
        ],
        "inbound": [True, False, True, False, True, False, False, True],
        "created_at": pd.to_datetime([
            "2023-01-01 10:00", "2023-01-01 10:05",
            "2023-01-01 10:10", "2023-01-01 10:15",
            "2023-01-01 11:00", "2023-01-01 11:05",
            "2023-01-01 12:00", "2023-01-01 13:00",
        ]),
        "text": [
            "@TestBrand my order hasn't arrived! https://t.co/abc",
            "@customer1 Sorry to hear that. Can you DM us your order number?",
            "@TestBrand I already sent it twice!!",
            "@customer1 Let me check on that for you.",
            "@TestBrand when will my refund be processed?",
            "@customer2 Refunds typically take 3-5 business days.",
            "@customer3 We can help with that.",
            "@OtherBrand why is my bill so high?",
        ],
        "response_tweet_id": ["2", "3", "4", None, "6", None, None, None],
        "in_response_to_tweet_id": [None, "1", "2", "3", None, "5", None, None],
    })


@pytest.fixture
def threaded_df() -> pd.DataFrame:
    """Create a sample DataFrame with conversation threads."""
    return pd.DataFrame({
        "conversation_id": [
            "conv_000000", "conv_000000", "conv_000000", "conv_000000",
            "conv_000001", "conv_000001",
        ],
        "tweet_id": ["1", "2", "3", "4", "5", "6"],
        "author_id": [
            "customer1", "TestBrand", "customer1", "TestBrand",
            "customer2", "TestBrand",
        ],
        "role": [
            "customer", "brand", "customer", "brand",
            "customer", "brand",
        ],
        "inbound": [True, False, True, False, True, False],
        "created_at": pd.to_datetime([
            "2023-01-01 10:00", "2023-01-01 10:05",
            "2023-01-01 10:10", "2023-01-01 10:15",
            "2023-01-01 11:00", "2023-01-01 11:05",
        ]),
        "text": [
            "my order hasn't arrived!",
            "Sorry to hear that. Can you DM us your order number?",
            "I already sent it twice!!",
            "Let me check on that for you.",
            "when will my refund be processed?",
            "Refunds typically take 3-5 business days.",
        ],
        "text_original": [
            "@TestBrand my order hasn't arrived! https://t.co/abc",
            "@customer1 Sorry to hear that. Can you DM us your order number?",
            "@TestBrand I already sent it twice!!",
            "@customer1 Let me check on that for you.",
            "@TestBrand when will my refund be processed?",
            "@customer2 Refunds typically take 3-5 business days.",
        ],
        "in_response_to_tweet_id": [None, "1", "2", "3", None, "5"],
        "response_tweet_id": ["2", "3", "4", None, "6", None],
    })


# ════════════════════════════════════════════════
#  Config Tests
# ════════════════════════════════════════════════


class TestConfig:
    """Tests for configuration loading."""

    def test_load_config_default(self) -> None:
        """Test loading the default configuration file."""
        from src.utils.config import load_config, DEFAULT_CONFIG_PATH

        if DEFAULT_CONFIG_PATH.exists():
            config = load_config()
            assert config.selected_brand != ""
            assert config.minimum_thread_length >= 1

    def test_config_has_required_fields(self, mock_config: Any) -> None:
        """Test that PipelineConfig has all required fields."""
        assert hasattr(mock_config, "selected_brand")
        assert hasattr(mock_config, "raw_data_path")
        assert hasattr(mock_config, "processed_dir")
        assert hasattr(mock_config, "minimum_thread_length")
        assert hasattr(mock_config, "lowercase")
        assert hasattr(mock_config, "remove_urls")
        assert hasattr(mock_config, "remove_duplicates")

    def test_config_brand_not_empty(self, mock_config: Any) -> None:
        """Test that brand is configured."""
        assert mock_config.selected_brand == "TestBrand"
        assert mock_config.brand_handle == "@TestBrand"

    def test_config_missing_file(self) -> None:
        """Test that loading a nonexistent config exits."""
        from src.utils.config import load_config

        with pytest.raises(SystemExit):
            load_config("nonexistent_config.yaml")


# ════════════════════════════════════════════════
#  Data Loader Tests
# ════════════════════════════════════════════════


class TestDataLoader:
    """Tests for the data loading module."""

    def test_expected_columns_defined(self) -> None:
        """Test that expected columns constant is defined."""
        from src.preprocessing.data_loader import EXPECTED_COLUMNS

        assert "tweet_id" in EXPECTED_COLUMNS
        assert "author_id" in EXPECTED_COLUMNS
        assert "text" in EXPECTED_COLUMNS
        assert "inbound" in EXPECTED_COLUMNS
        assert "in_response_to_tweet_id" in EXPECTED_COLUMNS

    def test_load_missing_file(self, mock_config: Any) -> None:
        """Test that loading nonexistent CSV exits gracefully."""
        from src.preprocessing.data_loader import load_raw_data

        mock_config.raw_data_path = Path("nonexistent.csv")
        with pytest.raises(SystemExit):
            load_raw_data(mock_config)


# ════════════════════════════════════════════════
#  Text Cleaning Tests
# ════════════════════════════════════════════════


class TestTextCleaner:
    """Tests for the text cleaning module."""

    def test_url_removal(self, mock_config: Any) -> None:
        """Test that URLs are removed from text."""
        from src.preprocessing.text_cleaner import _clean_single

        text = "Check this out https://example.com and http://test.org/page"
        result = _clean_single(text, mock_config)
        assert "https://" not in result
        assert "http://" not in result

    def test_whitespace_normalization(self, mock_config: Any) -> None:
        """Test that multiple spaces are collapsed."""
        from src.preprocessing.text_cleaner import _clean_single

        text = "hello    world   test"
        result = _clean_single(text, mock_config)
        assert "    " not in result
        assert "hello" in result

    def test_preserve_emojis(self, mock_config: Any) -> None:
        """Test that emojis are preserved during cleaning."""
        from src.preprocessing.text_cleaner import _clean_single

        text = "I love this! \U0001f60d\U0001f44d"
        result = _clean_single(text, mock_config)
        assert "\U0001f60d" in result
        assert "\U0001f44d" in result

    def test_leading_mention_removal(self, mock_config: Any) -> None:
        """Test that leading @mentions are stripped."""
        from src.preprocessing.text_cleaner import _clean_single

        text = "@TestBrand @user2 my order is late"
        result = _clean_single(text, mock_config)
        assert result.startswith("my") or result.startswith("My")

    def test_rt_prefix_removal(self, mock_config: Any) -> None:
        """Test that RT prefixes are removed."""
        from src.preprocessing.text_cleaner import _clean_single

        text = "RT @someuser: This is a retweet"
        result = _clean_single(text, mock_config)
        assert not result.startswith("RT")

    def test_lowercase_configurable(self, mock_config: Any) -> None:
        """Test that lowercase is configurable."""
        from src.preprocessing.text_cleaner import _clean_single

        text = "Hello World"

        # Default: no lowercase
        result = _clean_single(text, mock_config)
        assert "H" in result

        # Enable lowercase
        mock_config.lowercase = True
        result = _clean_single(text, mock_config)
        assert result == "hello world"

    def test_empty_text(self, mock_config: Any) -> None:
        """Test cleaning of empty text."""
        from src.preprocessing.text_cleaner import _clean_single

        assert _clean_single("", mock_config) == ""
        assert _clean_single("   ", mock_config) == ""

    def test_clean_text_preserves_original(self, mock_config: Any, threaded_df: pd.DataFrame) -> None:
        """Test that clean_text preserves original text."""
        from src.preprocessing.text_cleaner import clean_text

        result = clean_text(threaded_df, mock_config)
        assert "text_original" in result.columns


# ════════════════════════════════════════════════
#  Thread Reconstruction Tests
# ════════════════════════════════════════════════


class TestThreadBuilder:
    """Tests for thread reconstruction."""

    def test_build_threads_assigns_conversation_id(
        self, sample_tweets: pd.DataFrame, mock_config: Any
    ) -> None:
        """Test that conversation IDs are assigned."""
        from src.preprocessing.thread_builder import build_threads

        result = build_threads(sample_tweets, mock_config)
        if len(result) > 0:
            assert "conversation_id" in result.columns
            assert result["conversation_id"].notna().all()

    def test_build_threads_assigns_roles(
        self, sample_tweets: pd.DataFrame, mock_config: Any
    ) -> None:
        """Test that role column is added."""
        from src.preprocessing.thread_builder import build_threads

        result = build_threads(sample_tweets, mock_config)
        if len(result) > 0:
            assert "role" in result.columns
            assert set(result["role"].unique()).issubset({"customer", "brand"})

    def test_thread_chronological_order(
        self, sample_tweets: pd.DataFrame, mock_config: Any
    ) -> None:
        """Test that messages within a thread are chronological."""
        from src.preprocessing.thread_builder import build_threads

        result = build_threads(sample_tweets, mock_config)
        if len(result) > 0 and "created_at" in result.columns:
            for _, group in result.groupby("conversation_id"):
                timestamps = group["created_at"].dropna().tolist()
                assert timestamps == sorted(timestamps)


# ════════════════════════════════════════════════
#  Quality Filter Tests
# ════════════════════════════════════════════════


class TestQualityFilter:
    """Tests for quality filtering."""

    def test_removes_empty_text(
        self, mock_config: Any
    ) -> None:
        """Test that empty text messages are removed."""
        from src.preprocessing.quality_filter import apply_quality_filters

        df = pd.DataFrame({
            "conversation_id": ["c1", "c1", "c2", "c2"],
            "text": ["Hello", "", "How are you?", None],
            "author_id": ["a", "b", "c", "d"],
            "role": ["customer", "brand", "customer", "brand"],
        })

        result, counts = apply_quality_filters(df, mock_config)
        assert counts.get("empty_text", 0) >= 1
        assert len(result) < len(df)

    def test_removes_duplicates(
        self, mock_config: Any
    ) -> None:
        """Test that duplicate messages are removed."""
        from src.preprocessing.quality_filter import apply_quality_filters

        df = pd.DataFrame({
            "conversation_id": ["c1", "c1", "c1"],
            "text": ["Same message", "Same message", "Different"],
            "author_id": ["user1", "user1", "user2"],
            "role": ["customer", "customer", "brand"],
        })

        result, counts = apply_quality_filters(df, mock_config)
        assert counts.get("duplicates", 0) >= 1

    def test_removes_system_messages(
        self, mock_config: Any
    ) -> None:
        """Test that system/deleted messages are removed."""
        from src.preprocessing.quality_filter import apply_quality_filters

        df = pd.DataFrame({
            "conversation_id": ["c1", "c1", "c1"],
            "text": [
                "Normal message here",
                "This tweet is unavailable",
                "Another normal message",
            ],
            "author_id": ["a", "b", "c"],
            "role": ["customer", "brand", "customer"],
        })

        result, counts = apply_quality_filters(df, mock_config)
        assert counts.get("system_messages", 0) >= 1

    def test_removes_short_conversations(
        self, mock_config: Any
    ) -> None:
        """Test that conversations below minimum length are removed."""
        from src.preprocessing.quality_filter import apply_quality_filters

        df = pd.DataFrame({
            "conversation_id": ["c1", "c2", "c2"],
            "text": ["Solo message", "First message", "Second message"],
            "author_id": ["a", "b", "c"],
            "role": ["customer", "customer", "brand"],
        })
        mock_config.minimum_thread_length = 2

        result, counts = apply_quality_filters(df, mock_config)
        # c1 has only 1 message, should be removed
        assert "c1" not in result["conversation_id"].values if len(result) > 0 else True

    def test_returns_removal_counts(
        self, mock_config: Any
    ) -> None:
        """Test that removal counts are returned."""
        from src.preprocessing.quality_filter import apply_quality_filters

        df = pd.DataFrame({
            "conversation_id": ["c1", "c1"],
            "text": ["Hello world", "Hi there"],
            "author_id": ["a", "b"],
            "role": ["customer", "brand"],
        })

        result, counts = apply_quality_filters(df, mock_config)
        assert isinstance(counts, dict)


# ════════════════════════════════════════════════
#  Brand Filter Tests
# ════════════════════════════════════════════════


class TestBrandFilter:
    """Tests for brand filtering."""

    def test_filter_keeps_brand_tweets(
        self, sample_tweets: pd.DataFrame, mock_config: Any
    ) -> None:
        """Test that brand tweets are kept."""
        from src.preprocessing.brand_filter import filter_brand

        result = filter_brand(sample_tweets, mock_config)
        brand_tweets = result[result["author_id"] == "TestBrand"]
        assert len(brand_tweets) > 0

    def test_filter_removes_other_brands(
        self, sample_tweets: pd.DataFrame, mock_config: Any
    ) -> None:
        """Test that unrelated brand tweets are removed."""
        from src.preprocessing.brand_filter import filter_brand

        result = filter_brand(sample_tweets, mock_config)
        # OtherBrand standalone tweet (7) and its unrelated customer (8)
        # should not be in the filtered set
        other_brand = result[result["author_id"] == "OtherBrand"]
        assert len(other_brand) == 0

    def test_filter_invalid_brand_raises(
        self, sample_tweets: pd.DataFrame, mock_config: Any
    ) -> None:
        """Test that filtering for nonexistent brand raises."""
        from src.preprocessing.brand_filter import filter_brand

        mock_config.selected_brand = "NonExistentBrand123"
        with pytest.raises(ValueError):
            filter_brand(sample_tweets, mock_config)
