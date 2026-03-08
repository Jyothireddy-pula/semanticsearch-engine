"""Text preprocessing for noisy usenet/email style documents."""

from __future__ import annotations

import re

import pandas as pd

HEADER_PATTERN = re.compile(r"^(from|subject|organization|lines|reply-to|keywords|nntp-posting-host):", re.IGNORECASE)
PUNCTUATION_BLOAT_PATTERN = re.compile(r"[^\w\s.,!?;:'\"()-]")
WHITESPACE_PATTERN = re.compile(r"\s+")


class TextCleaner:
    """Applies deterministic cleaning rules for 20 Newsgroups data."""

    def __init__(self, min_words: int = 10) -> None:
        self.min_words = min_words

    def clean_document(self, text: str) -> str:
        """Clean a single document.

        Design decisions:
        - Remove header-like lines to avoid metadata leakage.
        - Drop quoted lines (starting with '>') to keep the author's content.
        - Truncate signatures after '-- ' marker (common in email clients).
        - Lowercase and normalize spaces for stable embedding behavior.
        - Remove unusual punctuation noise while preserving sentence punctuation.
        """
        lines = text.splitlines()
        body_lines = []
        in_header_block = True

        for line in lines:
            if in_header_block:
                if line.strip() == "":
                    in_header_block = False
                    continue
                if HEADER_PATTERN.match(line.strip()):
                    continue

            if line.strip().startswith(">"):
                continue
            body_lines.append(line)

        body = "\n".join(body_lines)

        signature_split = body.split("\n-- ", maxsplit=1)
        body = signature_split[0]

        body = body.lower()
        body = PUNCTUATION_BLOAT_PATTERN.sub(" ", body)
        body = WHITESPACE_PATTERN.sub(" ", body).strip()
        return body

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter dataframe into modeling-ready corpus."""
        cleaned = df.copy()
        cleaned["text"] = cleaned["raw_text"].fillna("").map(self.clean_document)
        cleaned["word_count"] = cleaned["text"].str.split().map(len)
        cleaned = cleaned[cleaned["word_count"] >= self.min_words].copy()
        cleaned = cleaned[["document_id", "text", "original_category"]].reset_index(drop=True)
        return cleaned
