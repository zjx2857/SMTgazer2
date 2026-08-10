#!/usr/bin/env python3
"""Build reusable Sibyl graph tensors from labeled SMT2 queries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from graph_cache import graph_cache_path, relative_instance_path


ROOT = Path(__file__).resolve().parent


def load_graph_builder():
    path = ROOT / "src" / "data_handlers" / "graph-builder.py"
    spec = importlib.util.spec_from_file_location("sibyl_graph_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Sibyl graph builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRAPH_BUILDER = load_graph_builder()


def build_one(task: tuple[str, str, str, bool]) -> tuple[str, str, float, str]:
    instance, query_root_text, cache_root_text, overwrite = task
    started = time.monotonic()
    try:
        relative = relative_instance_path(instance)
        source = Path(query_root_text) / relative
        target = graph_cache_path(cache_root_text, instance)
        if target.is_file() and target.stat().st_size > 0 and not overwrite:
            return instance, "cached", time.monotonic() - started, ""
        if not source.is_file():
            raise FileNotFoundError(source)

        nodes, edges, edge_attr = GRAPH_BUILDER.build_graph(str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                nodes=np.asarray(nodes),
                edges=np.asarray(edges),
                edge_attr=np.asarray(edge_attr),
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return instance, "built", time.monotonic() - started, ""
    except Exception as error:  # returned to the parent with the instance name
        return instance, "failed", time.monotonic() - started, repr(error)


def bounded_results(tasks, workers: int):
    iterator = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(max(1, workers * 2)):
            try:
                pending.add(executor.submit(build_one, next(iterator)))
            except StopIteration:
                break

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    pending.add(executor.submit(build_one, next(iterator)))
                except StopIteration:
                    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sibyl graph cache files")
    parser.add_argument("--query-root", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "test"), default=("train", "test")
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.query_root = args.query_root.expanduser().resolve()
    args.labels = args.labels.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.query_root.is_dir():
        parser.error(f"query root is not a directory: {args.query_root}")
    if not args.labels.is_file():
        parser.error(f"labels file does not exist: {args.labels}")
    return args


def main() -> None:
    args = parse_args()
    with args.labels.open("r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    instances: list[str] = []
    seen: set[str] = set()
    for split in args.splits:
        if split not in labels:
            raise KeyError(f"Missing {split!r} in {args.labels}")
        for instance in labels[split]:
            if instance not in seen:
                relative_instance_path(instance)
                seen.add(instance)
                instances.append(instance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(instances)
    counts = {"built": 0, "cached": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    started = time.monotonic()
    tasks = (
        (instance, str(args.query_root), str(args.output_dir), args.overwrite)
        for instance in instances
    )
    for completed, (instance, status, _duration, error) in enumerate(
        bounded_results(tasks, args.workers), start=1
    ):
        counts[status] += 1
        if error:
            failures.append((instance, error))
        if completed == total or completed % max(1, total // 100) == 0:
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            print(
                f"[GRAPH] {completed}/{total} built={counts['built']} "
                f"cached={counts['cached']} failed={counts['failed']} rate={rate:.2f}/s",
                flush=True,
            )

    manifest = {
        "labels": str(args.labels),
        "query_root": str(args.query_root),
        "output_dir": str(args.output_dir),
        "splits": list(args.splits),
        "workers": args.workers,
        "total": total,
        **counts,
        "duration_seconds": time.monotonic() - started,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[GRAPH] manifest={manifest_path}")
    if failures:
        for instance, error in failures[:20]:
            print(f"[GRAPH][ERROR] {instance}: {error}")
        raise RuntimeError(f"Failed to build {len(failures)}/{total} graph files")


if __name__ == "__main__":
    main()
