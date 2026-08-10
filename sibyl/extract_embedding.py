#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parent
NETWORK_DIR = ROOT / "src" / "networks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NETWORK_DIR))

from gnn import GAT
from graph_cache import CachedGraphDataset, graph_cache_path


def _load_graph_builder():
    path = ROOT / "src" / "data_handlers" / "graph-builder.py"
    spec = importlib.util.spec_from_file_location("sibyl_graph_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Sibyl graph builder from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRAPH_BUILDER = _load_graph_builder()


class SibylFeatureExtractor:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        checkpoint = torch.load(Path(model_path).expanduser(), map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            model_config = checkpoint.get("model_config")
        else:
            state_dict = checkpoint
            model_config = None
        if "fcLast.bias" not in state_dict:
            raise ValueError("The checkpoint is not a Sibyl GAT state dictionary")

        output_size = state_dict["fcLast.bias"].numel()
        if model_config is None:
            model_config = {
                "passes": 2,
                "inputLayerSize": 67,
                "outputLayerSize": output_size,
                "numAttentionLayers": 5,
                "mode": "cat",
                "pool": "attention",
                "k": 20,
                "dropout": 0,
                "shouldJump": True,
            }
        else:
            model_config = dict(model_config)
            if model_config.get("outputLayerSize") != output_size:
                raise ValueError("Checkpoint model metadata does not match its state dict")
        self.model = GAT(**model_config)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, query_path: str | Path) -> list[float]:
        nodes, edges, edge_attr = GRAPH_BUILDER.build_graph(
            str(Path(query_path).expanduser())
        )
        x = torch.as_tensor(nodes, dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(edges, dtype=torch.long, device=self.device)
        edge_features = torch.as_tensor(
            edge_attr, dtype=torch.float32, device=self.device
        )
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        with torch.inference_mode():
            embedding = self.model.encode_graph(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_features,
                batch=batch,
            )

        if tuple(embedding.shape) != (1, 201):
            raise ValueError(f"Expected a [1, 201] embedding, got {embedding.shape}")
        if not torch.isfinite(embedding).all().item():
            raise ValueError("Sibyl produced a non-finite embedding")

        return embedding[0].cpu().tolist()

    def has_cached_graph(self, instance: str, cache_root: str | Path) -> bool:
        return graph_cache_path(cache_root, instance).is_file()

    def iter_cached_batches(
        self,
        instances: list[str],
        cache_root: str | Path,
        batch_size: int = 8,
        workers: int = 4,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if workers < 0:
            raise ValueError("workers must be non-negative")

        dataset = CachedGraphDataset(instances, cache_root)
        loader_options = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
        }
        if workers:
            loader_options.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(dataset, **loader_options)
        with torch.inference_mode():
            for graphs in loader:
                sample_ids = graphs.sample_id.tolist()
                graphs = graphs.to(self.device, non_blocking=True)
                embeddings = self.model.encode_graph(
                    x=graphs.x,
                    edge_index=graphs.edge_index,
                    edge_attr=graphs.edge_attr,
                    batch=graphs.batch,
                )
                if embeddings.ndim != 2 or embeddings.size(1) != 201:
                    raise ValueError(
                        f"Expected [batch, 201] embeddings, got {embeddings.shape}"
                    )
                if not torch.isfinite(embeddings).all().item():
                    raise ValueError("Sibyl produced a non-finite embedding")
                yield sample_ids, embeddings.cpu().tolist()

    def extract_cached_many(
        self,
        instances: list[str],
        cache_root: str | Path,
        batch_size: int = 8,
        workers: int = 4,
    ) -> list[list[float]]:
        features: list[list[float] | None] = [None] * len(instances)
        for sample_ids, vectors in self.iter_cached_batches(
            instances, cache_root, batch_size, workers
        ):
            for sample_id, feature in zip(sample_ids, vectors):
                features[sample_id] = feature

        if any(feature is None for feature in features):
            raise RuntimeError("Batched Sibyl extraction did not return every input")
        return features  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a Sibyl GNN embedding")
    parser.add_argument("query", help="Path to an SMT2 query")
    parser.add_argument("--model", required=True, help="Path to a Sibyl checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    extractor = SibylFeatureExtractor(args.model, args.device)
    feature = extractor.extract(args.query)
    print(json.dumps(feature, allow_nan=False))


if __name__ == "__main__":
    main()
