"""Offline artifact builder for production semantic search serving."""

from __future__ import annotations

import argparse
import json

from search.semantic_search import build_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic search artifacts for API runtime")
    parser.add_argument("--dataset-root", required=True, help="Path to extracted 20 Newsgroups root")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output artifact directory")
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=20)
    args = parser.parse_args()

    manifest = build_artifacts(
        dataset_root=args.dataset_root,
        artifacts_dir=args.artifacts_dir,
        min_words=args.min_words,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
