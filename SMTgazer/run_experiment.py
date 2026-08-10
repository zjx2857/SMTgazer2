#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "machfea" / "infer_result"
OUTPUT_DIR = ROOT / "output"
DEFAULT_CLUSTERS = {
    "Equality+LinearArith": 2,
    "SyGuS": 3,
}
DATASETS = (
    "BMC",
    "Equality+LinearArith",
    "QF_Bitvec",
    "QF_LinearRealArith",
    "QF_NonLinearIntArith",
    "SyGuS",
    "SymEx",
)
MACHFEA_DIR = ROOT / "machfea"
sys.path.insert(0, str(MACHFEA_DIR))

from mach_run_inference import feature_key as encode_feature_key


def set_gpu_phase(phase: str) -> None:
    phase_file = os.environ.get("SMTGAZER_GPU_PHASE_FILE")
    if phase_file:
        Path(phase_file).write_text(phase + "\n", encoding="utf-8")
    print(f"[PHASE] {phase}", flush=True)


def run_command(command: list[str], environment: dict[str, str], dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


def load_experiment_inputs(dataset: str) -> tuple[dict, list[str], Path, Path]:
    labels_path = ROOT / "data" / f"{dataset}Labels.json"
    solver_path = ROOT / "machfea" / f"{dataset}_solver.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    if not solver_path.is_file():
        raise FileNotFoundError(f"Solver file not found: {solver_path}")

    with labels_path.open("r", encoding="utf-8") as label_file:
        labels = json.load(label_file)
    with solver_path.open("r", encoding="utf-8") as solver_file:
        solver_list = json.load(solver_file)["solver_list"]

    for split in ("train", "test"):
        if split not in labels:
            raise ValueError(f"Missing {split!r} split in {labels_path}")
        for instance, times in labels[split].items():
            if len(times) != len(solver_list):
                raise ValueError(
                    f"Solver count mismatch for {split} instance {instance}: "
                    f"{len(times)} labels, {len(solver_list)} solvers"
                )

    return labels, solver_list, labels_path, solver_path


def eligible_instances(labels: dict, split: str) -> set[str]:
    return {
        instance
        for instance, times in labels[split].items()
        if times and min(float(value) for value in times) < 2400
    }


def expected_feature_keys(dataset: str, labels: dict, split: str) -> set[str]:
    return {
        encode_feature_key(dataset, instance)
        for instance in eligible_instances(labels, split)
    }


def validate_feature_file(
    path: Path,
    expected_keys: set[str],
    allow_partial: bool,
) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")
    with path.open("r", encoding="utf-8") as feature_file:
        features = json.load(feature_file)
    if not isinstance(features, dict) or not features:
        raise ValueError(f"Feature file is empty or invalid: {path}")

    for key, vector in features.items():
        if not isinstance(vector, list) or len(vector) != 201:
            raise ValueError(f"Feature {key!r} in {path} is not 201 dimensional")
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(f"Feature {key!r} in {path} contains non-finite values")

    actual_keys = set(features)
    if allow_partial:
        unexpected = actual_keys - expected_keys
        if unexpected:
            raise ValueError(
                f"Feature file contains unexpected keys in {path}: "
                f"{sorted(unexpected)[:3]}"
            )
    elif actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        raise ValueError(
            f"Feature key mismatch in {path}: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    return actual_keys


def canonical_feature_paths(dataset: str) -> tuple[Path, Path]:
    return (
        FEATURE_DIR / f"{dataset}_train_feature.json",
        FEATURE_DIR / f"{dataset}_test_feature.json",
    )


def validate_features(
    dataset: str,
    labels: dict,
    allow_partial: bool,
) -> dict[str, set[str]]:
    validated = {}
    for split, path in zip(("train", "test"), canonical_feature_paths(dataset)):
        expected = expected_feature_keys(dataset, labels, split)
        actual = validate_feature_file(path, expected, allow_partial)
        print(f"[OK] {split} features: {len(actual)}/{len(expected)}")
        validated[split] = actual
    return validated


def generate_features(
    args: argparse.Namespace,
    labels: dict,
    labels_path: Path,
    environment: dict[str, str],
) -> None:
    targets = canonical_feature_paths(args.dataset)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Feature output exists; use --overwrite: {joined}")

    if args.dry_run:
        staging = ROOT / "tmp" / f"feature_stage_{args.dataset}"
        command = feature_command(args, labels_path, staging)
        run_command(command, environment, True)
        target_list = ", ".join(str(path) for path in targets)
        print(
            "[DRY-RUN] validate staged 201-dimensional features and replace "
            f"{target_list}"
        )
        return

    temp_root = ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"feature_stage_{args.dataset}_", dir=temp_root
    ) as directory:
        staging = Path(directory)
        run_command(feature_command(args, labels_path, staging), environment, False)

        staged_paths = (
            staging / f"{args.dataset}_train_feature.json",
            staging / f"{args.dataset}_test_feature.json",
        )
        for split, path in zip(("train", "test"), staged_paths):
            expected = expected_feature_keys(args.dataset, labels, split)
            actual = validate_feature_file(
                path, expected, args.allow_partial_features
            )
            print(f"[OK] staged {split} features: {len(actual)}/{len(expected)}")

        for staged, target in zip(staged_paths, targets):
            os.replace(staged, target)
            print(f"[OK] installed {target}")


def feature_command(
    args: argparse.Namespace,
    labels_path: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "machfea/mach_run_inference.py",
        "--dataset",
        args.dataset,
        "--labels",
        str(labels_path),
        "--query-root",
        str(args.query_root),
        "--sibyl-root",
        str(args.sibyl_root),
        "--model",
        str(args.checkpoint),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
    ]
    if args.sibyl_graph_cache is not None:
        command.extend(
            [
                "--graph-cache",
                str(args.sibyl_graph_cache),
                "--batch-size",
                str(args.feature_batch_size),
                "--workers",
                str(args.gnn_workers),
            ]
        )
    return command


def train_sibyl(
    args: argparse.Namespace,
    environment: dict[str, str],
) -> None:
    set_gpu_phase("sibyl_graph_build")
    graph_command = [
        sys.executable,
        "-u",
        str(args.sibyl_root / "prepare_graph_cache.py"),
        "--query-root",
        str(args.query_root),
        "--labels",
        str(args.sibyl_train_labels),
        "--output-dir",
        str(args.sibyl_graph_cache),
        "--splits",
        "train",
        "test",
        "--workers",
        str(args.graph_workers),
    ]
    run_command(graph_command, environment, args.dry_run)

    set_gpu_phase("sibyl_gnn_train")
    metrics_path = args.sibyl_checkpoint_output.with_suffix(".metrics.json")
    train_command = [
        sys.executable,
        "-u",
        str(args.sibyl_root / "train_gnn.py"),
        "--data",
        str(args.sibyl_graph_cache),
        "--labels",
        str(args.sibyl_train_labels),
        "--output",
        str(args.sibyl_checkpoint_output),
        "--metrics",
        str(metrics_path),
        "--device",
        args.device,
        "--epochs",
        str(args.gnn_epochs),
        "--batch-size",
        str(args.gnn_batch_size),
        "--workers",
        str(args.gnn_workers),
        "--seed",
        str(args.gnn_seed),
    ]
    run_command(train_command, environment, args.dry_run)
    if not args.dry_run and not args.sibyl_checkpoint_output.is_file():
        raise FileNotFoundError(
            f"Sibyl training did not create {args.sibyl_checkpoint_output}"
        )
    args.checkpoint = args.sibyl_checkpoint_output
    print(f"[OK] using newly trained Sibyl checkpoint: {args.checkpoint}")


def validate_train_result(
    path: Path,
    cluster_count: int,
    solver_count: int,
) -> int:
    with path.open("r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    if set(result) != {"portfolio", "lim", "center"}:
        raise ValueError(f"Invalid training result keys: {path}")

    centers = result["center"]["center"]
    if not 1 <= len(centers) <= cluster_count:
        raise ValueError(
            f"Expected 1..{cluster_count} centers, got {len(centers)}"
        )
    if any(
        len(center) != 201
        or not all(math.isfinite(float(value)) for value in center)
        for center in centers
    ):
        raise ValueError(f"Training centers are not 201 dimensional: {path}")
    minimum = result["lim"]["min"]
    scale = result["lim"]["sub"]
    if len(minimum) != 201 or len(scale) != 201:
        raise ValueError(f"Training normalization is not 201 dimensional: {path}")
    if not all(math.isfinite(float(value)) for value in minimum):
        raise ValueError(f"Training normalization minimum is not finite: {path}")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in scale):
        raise ValueError(f"Training normalization scale is invalid: {path}")
    if len(result["portfolio"]) != len(centers):
        raise ValueError(f"Training portfolio count mismatch: {path}")
    expected_clusters = {str(index) for index in range(len(centers))}
    if set(result["portfolio"]) != expected_clusters:
        raise ValueError(f"Training portfolio cluster keys are invalid: {path}")

    for cluster, value in result["portfolio"].items():
        solvers, time_slices = value
        if len(solvers) != 4 or len(time_slices) != 4:
            raise ValueError(f"Invalid portfolio for cluster {cluster} in {path}")
        if any(not 0 <= int(index) < solver_count for index in solvers):
            raise ValueError(f"Invalid solver index for cluster {cluster} in {path}")
        if not all(math.isfinite(float(value)) for value in time_slices):
            raise ValueError(f"Non-finite time slice for cluster {cluster} in {path}")
    return len(centers)


def validate_test_result(
    path: Path,
    expected_keys: set[str],
    solver_list: list[str],
) -> None:
    with path.open("r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    if set(result) != expected_keys:
        raise ValueError(f"Test result key mismatch in {path}")
    for key, value in result.items():
        solvers, time_slices = value
        if len(solvers) != 4 or len(time_slices) != 4:
            raise ValueError(f"Invalid test portfolio for {key!r} in {path}")
        if any(solver not in solver_list for solver in solvers):
            raise ValueError(f"Unknown solver in test portfolio for {key!r} in {path}")
        if not all(math.isfinite(float(item)) for item in time_slices):
            raise ValueError(f"Non-finite test time slice for {key!r} in {path}")


def run_portfolios(
    args: argparse.Namespace,
    labels: dict,
    solver_list: list[str],
    solver_path: Path,
    cluster_count: int,
    environment: dict[str, str],
) -> None:
    set_gpu_phase("portfolio")
    if args.dry_run and args.stage == "all":
        test_feature_keys = expected_feature_keys(args.dataset, labels, "test")
    else:
        validated = validate_features(
            args.dataset, labels, args.allow_partial_features
        )
        test_feature_keys = validated["test"]
    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conflicts = []
    for seed in args.seeds:
        train_path = OUTPUT_DIR / (
            f"train_result_{args.dataset}_4_{cluster_count}_{seed}.json"
        )
        conflicts.extend([train_path] if train_path.exists() else [])
        conflicts.extend(
            path
            for center_count in range(1, cluster_count + 1)
            if (
                path := OUTPUT_DIR
                / f"test_result_{args.dataset}_{seed}_{center_count}.json"
            ).exists()
        )
    if conflicts and not args.overwrite:
        joined = ", ".join(str(path) for path in sorted(set(conflicts)))
        raise FileExistsError(f"Experiment output exists; use --overwrite: {joined}")

    for seed in args.seeds:
        train_path = OUTPUT_DIR / (
            f"train_result_{args.dataset}_4_{cluster_count}_{seed}.json"
        )
        if args.overwrite and not args.dry_run:
            train_path.unlink(missing_ok=True)

        train_command = [
            sys.executable,
            "-u",
            "SMTportfolio.py",
            "train",
            "-dataset",
            args.dataset,
            "-solverdict",
            str(solver_path.relative_to(ROOT)),
            "-seed",
            str(seed),
            "-cluster_num",
            str(cluster_count),
        ]
        run_command(train_command, environment, args.dry_run)

        actual_centers = None
        if not args.dry_run:
            if not train_path.is_file():
                raise FileNotFoundError(f"Training did not create {train_path}")
            actual_centers = validate_train_result(
                train_path, cluster_count, len(solver_list)
            )
            print(f"[OK] training result {train_path}")

        infer_command = [
            sys.executable,
            "-u",
            "SMTportfolio.py",
            "infer",
            "-clusterPortfolio",
            str(train_path.relative_to(ROOT)),
            "-dataset",
            args.dataset,
            "-solverdict",
            str(solver_path.relative_to(ROOT)),
            "-seed",
            str(seed),
        ]
        if args.dry_run:
            run_command(infer_command, environment, True)
            print(
                f"[DRY-RUN] expected test result: output/test_result_"
                f"{args.dataset}_{seed}_<1..{cluster_count}>.json"
            )
        else:
            test_path = OUTPUT_DIR / (
                f"test_result_{args.dataset}_{seed}_{actual_centers}.json"
            )
            if test_path.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        f"Test output exists; use --overwrite: {test_path}"
                    )
                test_path.unlink()
            run_command(infer_command, environment, False)
            if not test_path.is_file():
                raise FileNotFoundError(f"Inference did not create {test_path}")
            validate_test_result(test_path, test_feature_keys, solver_list)
            print(f"[OK] test result {test_path}")


def validate_query_root(
    query_root: Path,
    labels: dict,
    allow_partial: bool,
) -> None:
    for split in ("train", "test"):
        eligible = eligible_instances(labels, split)
        missing = [
            instance
            for instance in eligible
            if not (query_root / instance.lstrip("/\\")).is_file()
        ]
        found = len(eligible) - len(missing)
        if found == 0:
            raise FileNotFoundError(
                f"No eligible {split} queries were found below {query_root}"
            )
        if missing and not allow_partial:
            raise FileNotFoundError(
                f"Missing {len(missing)}/{len(eligible)} eligible {split} queries "
                f"below {query_root}; first missing: {missing[0]}"
            )
        if missing:
            print(
                f"[WARN] {split} query coverage: {found}/{len(eligible)}",
                file=sys.stderr,
            )
        else:
            print(f"[OK] {split} query coverage: {found}/{len(eligible)}")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SMTgazer dataset experiment")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--stage",
        choices=("all", "features", "portfolio"),
        default="all",
    )
    parser.add_argument("--query-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sibyl-root", type=Path, default=ROOT.parent / "sibyl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--train-sibyl",
        action="store_true",
        help="Train a new Sibyl GNN before generating features",
    )
    parser.add_argument("--sibyl-train-labels", type=Path)
    parser.add_argument("--sibyl-graph-cache", type=Path)
    parser.add_argument("--sibyl-checkpoint-output", type=Path)
    parser.add_argument("--gnn-epochs", type=int, default=25)
    parser.add_argument("--gnn-batch-size", type=int, default=8)
    parser.add_argument("--gnn-workers", type=int, default=4)
    parser.add_argument("--graph-workers", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--gnn-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique non-negative integers")
    if args.clusters is not None and args.clusters < 1:
        parser.error("clusters must be a positive integer")
    for name in ("gnn_epochs", "gnn_batch_size", "graph_workers", "feature_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gnn_workers < 0:
        parser.error("--gnn-workers must be non-negative")
    if args.gnn_seed < 0:
        parser.error("--gnn-seed must be non-negative")
    if args.train_sibyl and args.stage == "portfolio":
        parser.error("--train-sibyl requires stage all or features")

    if args.stage in ("all", "features"):
        if args.query_root is None:
            parser.error("--query-root is required for feature generation")
        if not args.train_sibyl and args.checkpoint is None:
            parser.error("--checkpoint is required unless --train-sibyl is active")
        args.query_root = args.query_root.expanduser().resolve()
        args.sibyl_root = args.sibyl_root.expanduser().resolve()
        if not args.query_root.is_dir():
            parser.error(f"query root is not a directory: {args.query_root}")
        if not (args.sibyl_root / "extract_embedding.py").is_file():
            parser.error(f"invalid Sibyl root: {args.sibyl_root}")
        if args.train_sibyl:
            if args.checkpoint is not None:
                parser.error("do not combine --train-sibyl with --checkpoint")
            for required in ("prepare_graph_cache.py", "train_gnn.py", "graph_cache.py"):
                if not (args.sibyl_root / required).is_file():
                    parser.error(f"Sibyl training file is missing: {required}")
            args.sibyl_train_labels = (
                args.sibyl_train_labels
                or args.sibyl_root / "data" / f"{args.dataset}_GNN_Labels.json"
            ).expanduser().resolve()
            if not args.sibyl_train_labels.is_file():
                parser.error(f"Sibyl training labels not found: {args.sibyl_train_labels}")
            args.sibyl_graph_cache = (
                args.sibyl_graph_cache
                or ROOT.parent / ".cache" / "sibyl_graphs" / args.dataset
            ).expanduser().resolve()
            if args.sibyl_checkpoint_output is None:
                log_dir = Path(
                    os.environ.get("SMTGAZER_LOG_DIR", ROOT.parent / "artifacts")
                )
                args.sibyl_checkpoint_output = (
                    log_dir
                    / "checkpoints"
                    / f"{args.dataset}_gnn_seed{args.gnn_seed}.pt"
                )
            args.sibyl_checkpoint_output = (
                args.sibyl_checkpoint_output.expanduser().resolve()
            )
            # The upstream GNN label file intentionally omits unparsable queries.
            args.allow_partial_features = True
        else:
            args.checkpoint = args.checkpoint.expanduser().resolve()
            if not args.checkpoint.is_file():
                parser.error(f"checkpoint not found: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    labels, solver_list, labels_path, solver_path = load_experiment_inputs(args.dataset)
    cluster_count = args.clusters or DEFAULT_CLUSTERS.get(args.dataset, 20)
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    environment["PYTHONUNBUFFERED"] = "1"

    if args.stage in ("all", "features"):
        if args.train_sibyl:
            train_sibyl(args, environment)
        validate_query_root(
            args.query_root, labels, args.allow_partial_features
        )
        if args.dry_run:
            print(f"[DRY-RUN] expected checkpoint: {args.checkpoint}")
        else:
            print(
                f"[INFO] checkpoint sha256: {checkpoint_sha256(args.checkpoint)}"
            )
        set_gpu_phase("sibyl_feature_extract")
        generate_features(args, labels, labels_path, environment)
    if args.stage in ("all", "portfolio"):
        run_portfolios(
            args,
            labels,
            solver_list,
            solver_path,
            cluster_count,
            environment,
        )
    if args.dry_run:
        print("[DRY-RUN] experiment commands validated")
    else:
        print("[OK] experiment stage completed")


if __name__ == "__main__":
    main()
