"""Semantic retrieval module for historical support conversations.

Constructs an index over historical resolved customer-brand conversation pairs,
embeds customer queries, computes cosine similarity, and retrieves top-k evidence items.
Provides historical context for reply generation and escalation reasoning without generating text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Default sample evidence dataset if historical data is not yet generated
DEFAULT_HISTORICAL_EVIDENCE = [
    {
        "conversation_id": "conv_hist_001",
        "customer_text": "Where is my order? Order #10293 has not arrived yet.",
        "brand_reply": "Please DM us your order number and full name so we can trace your shipment details.",
    },
    {
        "conversation_id": "conv_hist_002",
        "customer_text": "I want to cancel order #4092 immediately.",
        "brand_reply": "We can assist with cancellations if the item has not shipped yet. Please DM us your order details.",
    },
    {
        "conversation_id": "conv_hist_003",
        "customer_text": "My package was damaged upon delivery, box was crushed.",
        "brand_reply": "We apologize for the damaged package. Please DM us your order ID so we can send a replacement.",
    },
    {
        "conversation_id": "conv_hist_004",
        "customer_text": "I was charged twice for my order on my credit card.",
        "brand_reply": "We apologize for the billing error. Please reach out via DM with your order number and transaction details so we can investigate and process a refund.",
    },
    {
        "conversation_id": "conv_hist_005",
        "customer_text": "I can't log into my account, password reset isn't working.",
        "brand_reply": "For security reasons regarding account access, please visit our online account recovery help center.",
    },
]


@dataclass
class RetrievedItem:
    """A single historical retrieved conversation evidence item.

    Attributes:
        conversation_id: Unique thread identifier of the historical example.
        customer_text: Customer opening message.
        brand_reply: Resolution reply provided by the brand.
        similarity_score: Cosine similarity score to the query (0.0 to 1.0).
    """

    conversation_id: str
    customer_text: str
    brand_reply: str
    similarity_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conversation_id": self.conversation_id,
            "customer_text": self.customer_text,
            "brand_reply": self.brand_reply,
            "similarity_score": round(self.similarity_score, 4),
        }


@dataclass
class RetrievedEvidence:
    """Collection of retrieved evidence items for a query.

    Attributes:
        query: Original customer query text.
        items: List of retrieved evidence items ordered by similarity.
    """

    query: str
    items: list[RetrievedItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "items": [item.to_dict() for item in self.items],
        }


class SemanticRetriever:
    """Semantic retrieval engine over historical customer support conversations.

    Uses sentence-transformers when available, falling back to TF-IDF cosine
    similarity for lightweight offline execution.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the semantic retriever.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.top_k = self.config.agent_top_k
        self.df: pd.DataFrame | None = None
        self.embeddings: np.ndarray | None = None
        self.use_st = False
        self.st_model: Any = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None
        self.is_indexed = False

        # Try initializing SentenceTransformer if requested & available
        self._init_encoder()

    def _init_encoder(self) -> None:
        """Attempt to load SentenceTransformer model if installed."""
        model_name = self.config.agent_embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer model '{model_name}'...")
            self.st_model = SentenceTransformer(model_name)
            self.use_st = True
            logger.info("SentenceTransformer encoder loaded successfully.")
        except Exception as e:
            logger.info(
                f"SentenceTransformer not available or model load failed ({e}). "
                "Using TF-IDF cosine similarity fallback for retrieval."
            )
            self.use_st = False

    def build_index(self, df: pd.DataFrame | None = None) -> None:
        """Build the search index over historical customer support conversations.

        Args:
            df: Optional DataFrame with conversation_id, opening_message/customer_text, and full_conversation/brand_reply.
        """
        if df is None:
            df = self._load_historical_data()

        self.df = df.copy()

        # Ensure uniform column names
        if "opening_message" in self.df.columns and "customer_text" not in self.df.columns:
            self.df["customer_text"] = self.df["opening_message"]

        if "full_conversation" in self.df.columns and "brand_reply" not in self.df.columns:
            # Extract brand reply from full_conversation text
            self.df["brand_reply"] = self.df["full_conversation"].apply(self._extract_brand_reply)

        texts = self.df["customer_text"].fillna("").astype(str).tolist()

        logger.info(f"Indexing {len(texts)} historical conversations for retrieval...")

        if self.use_st and self.st_model is not None:
            self.embeddings = self.st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        self.is_indexed = True
        logger.info("SemanticRetriever index built successfully.")

    def _extract_brand_reply(self, full_conv: str) -> str:
        """Extract first brand reply from full conversation text."""
        if not full_conv or pd.isna(full_conv):
            return "Please DM us so we can assist you."

        lines = str(full_conv).split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("[BRAND]"):
                return line.replace("[BRAND]", "").strip()
            if "AmazonHelp" in line:
                return line.strip()

        return "Please DM us your details so we can investigate."

    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data from configured paths or default evidence."""
        index_source = self.config.agent_index_source
        if index_source.exists():
            try:
                df = pd.read_csv(index_source)
                if len(df) > 0 and "conversation_id" in df.columns:
                    return df
            except Exception as e:
                logger.warning(f"Could not load index source from {index_source}: {e}")

        # Try processed_conversations.csv
        proc_path = self.config.processed_conversations_path
        if proc_path.exists():
            try:
                df = pd.read_csv(proc_path)
                cust_df = df[df["role"] == "customer"].groupby("conversation_id").first().reset_index()
                brand_df = df[df["role"] == "brand"].groupby("conversation_id").first().reset_index()
                merged = pd.merge(cust_df, brand_df, on="conversation_id", suffixes=("_cust", "_brand"))
                if len(merged) > 0:
                    merged["customer_text"] = merged["text_cust"]
                    merged["brand_reply"] = merged["text_brand"]
                    return merged
            except Exception as e:
                logger.warning(f"Could not load processed conversations from {proc_path}: {e}")

        logger.info("Using default historical evidence dataset for retriever...")
        return pd.DataFrame(DEFAULT_HISTORICAL_EVIDENCE)

    def retrieve(self, query_text: str, top_k: int | None = None) -> RetrievedEvidence:
        """Retrieve top-k similar historical evidence items for an input query.

        Args:
            query_text: Input customer query.
            top_k: Optional number of items to retrieve (defaults to config value).

        Returns:
            RetrievedEvidence object containing query and retrieved items.
        """
        if not self.is_indexed:
            self.build_index()

        k = top_k or self.top_k
        clean_query = query_text.strip()

        if not clean_query or self.df is None or len(self.df) == 0:
            return RetrievedEvidence(query=query_text, items=[])

        if self.use_st and self.st_model is not None and self.embeddings is not None:
            query_emb = self.st_model.encode([clean_query], normalize_embeddings=True)
            sims = np.dot(self.embeddings, query_emb.T).squeeze()
            if np.isscalar(sims):
                sims = np.array([sims])
        elif self.vectorizer is not None and self.tfidf_matrix is not None:
            query_vec = self.vectorizer.transform([clean_query])
            sims = cosine_similarity(self.tfidf_matrix, query_vec).squeeze()
            if np.isscalar(sims):
                sims = np.array([sims])
        else:
            return RetrievedEvidence(query=query_text, items=[])

        # Get top-k indices
        k = min(k, len(sims))
        top_indices = np.argsort(sims)[::-1][:k]

        items: list[RetrievedItem] = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            items.append(
                RetrievedItem(
                    conversation_id=str(row.get("conversation_id", f"hist_{idx}")),
                    customer_text=str(row.get("customer_text", "")),
                    brand_reply=str(row.get("brand_reply", "")),
                    similarity_score=float(sims[idx]),
                )
            )

        return RetrievedEvidence(query=query_text, items=items)
