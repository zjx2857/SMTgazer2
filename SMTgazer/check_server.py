#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import subprocess
import sys
from importlib.metadata import version as distribution_version
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SIBYL_ROOT = ROOT.parent / "sibyl"
SUPPORTED_PYTHON = {(3, 10), (3, 11)}
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "torch": "2.2.0",
    "torch-geometric": "2.5.3",
    "pySMT": "0.9.6",
    "pyclustering": "0.10.1.2",
    "LightGBM": "4.5.0",
    "ConfigSpace": "1.2.1",
}


class LightGBMLogCapture:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(str(message))
        print(message)

    def warning(self, message: str) -> None:
        self.messages.append(str(message))
        print(message, file=sys.stderr)


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def require_version(name: str, actual: str) -> None:
    expected = EXPECTED_VERSIONS[name]
    if public_version(actual) != expected:
        raise RuntimeError(f"{name} must be {expected}, found {actual}")
    print(f"[OK] {name} {actual}")


def opencl_gpu_names() -> list[str]:
    try:
        result = subprocess.run(
            ["clinfo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError("clinfo is not installed") from error

    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"clinfo failed with exit code {result.returncode}{suffix}")

    device_matches = list(
        re.finditer(r"^\s*Device Name\s+(.+?)\s*$", output, re.MULTILINE)
    )
    gpu_names = []
    for index, match in enumerate(device_matches):
        block_end = (
            device_matches[index + 1].start()
            if index + 1 < len(device_matches)
            else len(output)
        )
        block = output[match.end() : block_end]
        device_type = re.search(
            r"^\s*Device Type\s+(.+?)\s*$", block, re.MULTILINE
        )
        if device_type and re.search(
            r"(?:^|[^A-Za-z0-9])GPU(?:$|[^A-Za-z0-9])",
            device_type.group(1),
            re.IGNORECASE,
        ):
            gpu_names.append(match.group(1).strip())

    if not gpu_names:
        raise RuntimeError("clinfo did not report an OpenCL device with type GPU")
    unique_names = sorted(set(gpu_names))
    print(f"[OK] OpenCL GPU device(s): {', '.join(unique_names)}")
    return unique_names


def validate_selected_gpu(messages: list[str], gpu_names: list[str]) -> str:
    log_text = "\n".join(messages)
    selected = re.findall(r"Using GPU Device:\s*([^,\r\n]+)", log_text)
    if not selected:
        raise RuntimeError("LightGBM did not report its selected OpenCL device")

    selected_name = selected[-1].strip()
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    normalized_selected = normalize(selected_name)
    if not normalized_selected or not any(
        normalized_selected in normalize(name) or normalize(name) in normalized_selected
        for name in gpu_names
    ):
        raise RuntimeError(
            f"LightGBM selected {selected_name!r}, which is not a clinfo GPU device"
        )
    return selected_name


def load_sibyl_extractor():
    module_path = SIBYL_ROOT / "extract_embedding.py"
    spec = importlib.util.spec_from_file_location("sibyl_extract_embedding", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Sibyl extractor from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SibylFeatureExtractor


def check_environment(torch_device: str, lightgbm_device_type: str) -> None:
    python_version = sys.version_info[:2]
    if python_version not in SUPPORTED_PYTHON:
        raise RuntimeError(
            f"Python must be 3.10 or 3.11, found {sys.version.split()[0]}"
        )
    print(f"[OK] Python {sys.version.split()[0]}")

    if lightgbm_device_type not in {"gpu", "cuda"}:
        raise RuntimeError(
            "LightGBM device type must be 'gpu' (OpenCL) or 'cuda', "
            f"found {lightgbm_device_type!r}"
        )
    os.environ["SMTGAZER_LIGHTGBM_DEVICE_TYPE"] = lightgbm_device_type

    import ConfigSpace
    import lightgbm
    import numpy as np
    import pysmt
    import smac
    import SMTportfolio
    import torch
    import torch_geometric
    from ConfigSpace import ConfigurationSpace, Float
    from lightgbm import LGBMRegressor
    from pysmt.operators import ALL_TYPES
    from smac.model.xg.xg import XG

    require_version("numpy", np.__version__)
    require_version("torch", torch.__version__)
    require_version("torch-geometric", torch_geometric.__version__)
    require_version("pySMT", pysmt.__version__)
    require_version("pyclustering", distribution_version("pyclustering"))
    require_version("LightGBM", lightgbm.__version__)
    require_version("ConfigSpace", distribution_version("ConfigSpace"))

    smac_path = Path(smac.__file__).resolve()
    if ROOT / "smac" not in smac_path.parents:
        raise RuntimeError(f"SMAC is not the local customized package: {smac_path}")
    print(f"[OK] local SMAC {smac_path}")

    node_size = len(ALL_TYPES) + 1
    if node_size != 67:
        raise RuntimeError(f"Sibyl requires 67 node features, pySMT provides {node_size}")
    print("[OK] pySMT node feature size 67")

    device = torch.device(torch_device)
    if device.type != "cuda":
        raise RuntimeError(f"GNN device must be CUDA for server experiments: {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    torch.empty(1, device=device)
    print(f"[OK] PyTorch CUDA device: {torch.cuda.get_device_name(device)}")

    gpu_names = (
        opencl_gpu_names() if lightgbm_device_type == "gpu" else None
    )
    lightgbm_log = LightGBMLogCapture()
    lightgbm.register_logger(lightgbm_log)

    rng = np.random.RandomState(0)
    X = rng.rand(512, 4)
    y = rng.rand(512)
    lightgbm_model = LGBMRegressor(
        n_estimators=2,
        min_data_in_leaf=1,
        max_bin=63,
        device_type=lightgbm_device_type,
        verbose=1,
    )
    lightgbm_model.fit(X, y)
    prediction = lightgbm_model.predict(X[:8])
    if not np.isfinite(prediction).all():
        raise RuntimeError("LightGBM GPU prediction contains non-finite values")
    if lightgbm_model.booster_.params["device_type"] != lightgbm_device_type:
        raise RuntimeError(
            f"LightGBM did not use the {lightgbm_device_type!r} backend"
        )
    if lightgbm_device_type == "gpu":
        assert gpu_names is not None
        selected_gpu = validate_selected_gpu(lightgbm_log.messages, gpu_names)
        print(f"[OK] LightGBM OpenCL GPU fit/predict on {selected_gpu}")
    else:
        print("[OK] LightGBM CUDA GPU fit/predict")

    configspace = ConfigurationSpace(seed=0)
    configspace.add([Float(f"x{index}", (0.0, 1.0)) for index in range(4)])
    hybrid_model = XG(
        configspace=configspace,
        n_estimators=2,
        n_trees=4,
        min_data_in_leaf=1,
        seed=0,
    )
    hybrid_model.train(X, y.reshape(-1, 1))
    mean, variance = hybrid_model.predict(X[:8])
    if mean.shape != (8, 1) or variance.shape != (8, 1):
        raise RuntimeError(
            f"Unexpected hybrid prediction shapes: {mean.shape}, {variance.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise RuntimeError("Hybrid prediction contains non-finite values")
    if (
        hybrid_model.model.booster_.params.get("device_type")
        != lightgbm_device_type
    ):
        raise RuntimeError(
            "The customized XG model did not use its requested GPU backend"
        )
    print("[OK] SMAC LightGBM+RandomForest hybrid fit/predict")

    cluster_features = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
    )
    initial_centers = SMTportfolio.kmeans_plusplus_initializer(
        cluster_features, 2, random_state=0
    ).initialize()
    xmeans_model = SMTportfolio.xmeans(
        cluster_features,
        initial_centers=initial_centers,
        kmax=3,
        ccore=False,
        random_state=0,
    )
    xmeans_model.process()
    clusters = xmeans_model.get_clusters()
    if not 1 <= len(xmeans_model.get_centers()) <= 3:
        raise RuntimeError("X-means returned an invalid number of centers")
    if sum(len(cluster) for cluster in clusters) != len(cluster_features):
        raise RuntimeError("X-means did not assign every smoke-test point")
    print("[OK] pyclustering X-means fit")

    checkpoint = SIBYL_ROOT / "inference/BMC/model_checkpoints/BMC_model_0.pt"
    query = (
        SIBYL_ROOT
        / "inference/BMC/example_queries/Problem13_label15_reach_Query4.smt2"
    )
    extractor_class = load_sibyl_extractor()
    embedding = extractor_class(checkpoint, torch_device).extract(query)
    if len(embedding) != 201 or not all(math.isfinite(value) for value in embedding):
        raise RuntimeError("Sibyl embedding is not a finite 201-value vector")
    print("[OK] Sibyl 201-value GNN embedding")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an SMTgazer GPU server")
    parser.add_argument(
        "--torch-device",
        default="cuda:0",
        help="Device used for Sibyl GNN inference",
    )
    parser.add_argument(
        "--lightgbm-device-type",
        choices=("gpu", "cuda"),
        default=os.environ.get("SMTGAZER_LIGHTGBM_DEVICE_TYPE", "gpu"),
        help="LightGBM backend: OpenCL 'gpu' or NVIDIA 'cuda'",
    )
    args = parser.parse_args()

    try:
        check_environment(args.torch_device, args.lightgbm_device_type)
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("[OK] Server is ready for experiments")


if __name__ == "__main__":
    main()
