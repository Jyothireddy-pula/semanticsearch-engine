"""Dataset loader utilities for the 20 Newsgroups extracted folder structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class RawDocument:
    """Represents a single raw document before preprocessing."""

    document_id: str
    category: str
    text: str


class NewsgroupsDatasetLoader:
    """Loads documents from an extracted 20 Newsgroups directory.

    Expected structure:
        root/
          alt.atheism/
            49960
            51119
          comp.graphics/
            ...
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

    def load(self) -> pd.DataFrame:
        """Load all documents into a dataframe.

        Returns dataframe with columns:
            document_id, original_category, raw_text
        """
        records: List[RawDocument] = []

        for category_dir in sorted(self.dataset_root.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            for doc_path in sorted(category_dir.iterdir()):
                if not doc_path.is_file():
                    continue
                try:
                    raw_text = doc_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                document_id = f"{category}/{doc_path.name}"
                records.append(
                    RawDocument(document_id=document_id, category=category, text=raw_text)
                )

        df = pd.DataFrame(
            {
                "document_id": [r.document_id for r in records],
                "original_category": [r.category for r in records],
                "raw_text": [r.text for r in records],
            }
        )
        return df
