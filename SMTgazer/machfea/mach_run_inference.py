from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from get_feature import DEFAULT_SIBYL_ROOT, load_extractor


SMT_COMP_ALIASES = {
    "Equality+LinearArith": "ELA",
    "QF_Bitvec": "QFBV",
    "QF_NonLinearIntArith": "QFNIA",
}
SMT_COMP_DATASETS = set(SMT_COMP_ALIASES) | {"QF_LinearRealArith"}


def load_labels(path: Path, dataset: str) -> dict:
    with path.open("r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    if "train" in labels and "test" in labels:
        return labels
    if dataset not in labels:
        raise KeyError(f"Dataset {dataset!r} is not present in {path}")
    return labels[dataset]


def feature_key(dataset: str, instance: str) -> str:
    instance = instance.replace("/", "_")
    if dataset in SMT_COMP_DATASETS:
        directory = SMT_COMP_ALIASES.get(dataset, dataset)
        prefix = "_data_sibly_sibyl_data_Comp_non-incremental_"
    else:
        directory = dataset
        prefix = f"_data_sibly_sibyl_data_{dataset}_{dataset}_"
    return f"./infer_result/{directory}/{prefix}{instance}.json"


def is_solvable(times: list[int | float]) -> bool:
    return bool(times) and min(float(value) for value in times) < 2400


def extract_split(
    extractor,
    dataset: str,
    split: str,
    labels: dict,
    query_root: Path,
    output_dir: Path,
    graph_cache: Path | None,
    batch_size: int,
    workers: int,
) -> tuple[int, int, int]:
    features = {}
    eligible = 0
    skipped = 0
    failures = []
    cached_instances = []

    for instance, times in labels[split].items():
        if not is_solvable(times):
            skipped += 1
            continue
        eligible += 1

        query_path = query_root / instance.lstrip("/\\")
        if not query_path.is_file():
            failures.append((instance, f"query not found: {query_path}"))
            continue

        if graph_cache is not None:
            if extractor.has_cached_graph(instance, graph_cache):
                cached_instances.append(instance)
            else:
                failures.append((instance, "graph cache file not found"))
        else:
            try:
                features[feature_key(dataset, instance)] = extractor.extract(query_path)
            except Exception as error:
                failures.append((instance, str(error)))

    if cached_instances:
        try:
            batches = extractor.iter_cached_batches(
                cached_instances,
                graph_cache,
                batch_size=batch_size,
                workers=workers,
            )
            for sample_ids, vectors in batches:
                for sample_id, vector in zip(sample_ids, vectors):
                    instance = cached_instances[sample_id]
                    features[feature_key(dataset, instance)] = vector
        except Exception as error:
            raise RuntimeError(f"{split}: batched GPU feature extraction failed") from error

    if failures:
        print(f"{split}: {len(failures)} feature extractions failed", file=sys.stderr)
        for instance, error in failures:
            print(f"  {instance}: {error}", file=sys.stderr)

    if eligible and not features:
        raise RuntimeError(f"{split}: no features were extracted; output was not written")

    output_path = output_dir / f"{dataset}_{split}_feature.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(features, output_file, allow_nan=False)

    return len(features), skipped, len(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Sibyl features for SMTgazer")
    parser.add_argument("--dataset", required=True, help="Dataset name used by SMTgazer")
    parser.add_argument("--labels", required=True, type=Path, help="Dataset label JSON")
    parser.add_argument("--query-root", required=True, type=Path, help="Root of SMT2 queries")
    parser.add_argument("--model", required=True, type=Path, help="Sibyl checkpoint")
    parser.add_argument(
        "--sibyl-root",
        default=DEFAULT_SIBYL_ROOT,
        type=Path,
        help="Path to the local Sibyl clone",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--graph-cache",
        type=Path,
        help="Optional prebuilt graph cache for batched GPU inference",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent / "infer_result",
        type=Path,
        help="Feature output directory",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels.expanduser(), args.dataset)
    query_root = args.query_root.expanduser()
    if not query_root.is_dir():
        parser.error(f"query root is not a directory: {query_root}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    graph_cache = args.graph_cache.expanduser().resolve() if args.graph_cache else None
    if graph_cache is not None and not graph_cache.is_dir():
        parser.error(f"graph cache is not a directory: {graph_cache}")

    extractor = load_extractor(args.sibyl_root, args.model, args.device)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        extracted, skipped, failed = extract_split(
            extractor,
            args.dataset,
            split,
            labels,
            query_root,
            output_dir,
            graph_cache,
            args.batch_size,
            args.workers,
        )
        print(f"{split}: extracted={extracted}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
