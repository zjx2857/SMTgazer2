#!/usr/bin/env python3
"""Shared helpers for Sibyl's on-disk PyG graph cache."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def relative_instance_path(instance: str) -> Path:
    relative = Path(instance.lstrip("/\\"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid dataset instance path: {instance!r}")
    return relative


def graph_cache_path(cache_root: str | Path, instance: str) -> Path:
    relative = relative_instance_path(instance)
    return Path(cache_root) / relative.with_suffix(".npz")


def load_cached_graph(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"nodes", "edges", "edge_attr"}
        if not required.issubset(archive.files):
            raise ValueError(f"Invalid graph cache file: {path}")
        nodes = archive["nodes"]
        edges = archive["edges"]
        edge_attr = archive["edge_attr"]

    if nodes.ndim != 2 or nodes.shape[1] != 67:
        raise ValueError(f"Expected [N, 67] nodes in {path}, got {nodes.shape}")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"Expected [2, E] edges in {path}, got {edges.shape}")
    if edge_attr.ndim != 1 or edge_attr.shape[0] != edges.shape[1]:
        raise ValueError(f"Edge attribute mismatch in {path}")
    return nodes, edges, edge_attr


class CachedGraphDataset(Dataset):
    """Load graph tensors lazily so DataLoader workers can overlap disk and GPU work."""

    def __init__(self, instances: Sequence[str], cache_root: str | Path) -> None:
        self.instances = list(instances)
        self.cache_root = Path(cache_root)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, item: int) -> Data:
        instance = self.instances[item]
        path = graph_cache_path(self.cache_root, instance)
        nodes, edges, edge_attr = load_cached_graph(path)
        return Data(
            x=torch.as_tensor(nodes, dtype=torch.float32),
            edge_index=torch.as_tensor(edges, dtype=torch.long),
            edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
            problemType=torch.tensor(0, dtype=torch.float32),
            sample_id=torch.tensor(item, dtype=torch.long),
        )
