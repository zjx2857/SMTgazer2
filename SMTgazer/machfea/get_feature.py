from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


DEFAULT_SIBYL_ROOT = Path(__file__).resolve().parents[2] / "sibyl"


def load_extractor(
    sibyl_root: str | Path, model_path: str | Path, device: str = "cpu"
):
    module_path = Path(sibyl_root).expanduser() / "extract_embedding.py"
    spec = importlib.util.spec_from_file_location("sibyl_extract_embedding", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Sibyl feature extractor from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SibylFeatureExtractor(model_path, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one Sibyl GNN feature")
    parser.add_argument("query", help="Path to an SMT2 query")
    parser.add_argument("--model", required=True, help="Path to a Sibyl checkpoint")
    parser.add_argument(
        "--sibyl-root",
        default=str(DEFAULT_SIBYL_ROOT),
        help="Path to the local Sibyl clone",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    extractor = load_extractor(args.sibyl_root, args.model, args.device)
    feature = extractor.extract(args.query)
    print(json.dumps(feature, allow_nan=False))


if __name__ == "__main__":
    main()
