#!/usr/bin/env python3
"""Train a Sibyl GAT checkpoint with a GPU-oriented mini-batch pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "networks"))

from gnn import GAT
from graph_cache import CachedGraphDataset


class LabeledGraphDataset(CachedGraphDataset):
    def __init__(self, labeled_instances, cache_root: Path) -> None:
        self.labeled_instances = list(labeled_instances)
        super().__init__([instance for instance, _ in self.labeled_instances], cache_root)

    def __getitem__(self, item: int):
        graph = super().__getitem__(item)
        label = torch.tensor(self.labeled_instances[item][1], dtype=torch.float32)
        return graph, label


def ranking_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    order = labels.argsort(dim=1)
    ranked_scores = scores.gather(1, order)
    losses = []
    solver_count = ranked_scores.size(1)
    for better in range(solver_count):
        for worse in range(better + 1, solver_count):
            target = torch.ones_like(ranked_scores[:, better])
            losses.append(
                functional.margin_ranking_loss(
                    ranked_scores[:, better],
                    ranked_scores[:, worse],
                    target,
                    margin=margin * (worse - better),
                )
            )
    return torch.stack(losses).mean()


def loader_options(args: argparse.Namespace) -> dict:
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    if args.workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    return options


def make_train_weights(labeled_instances) -> list[float]:
    labels = np.asarray([values for _, values in labeled_instances])
    winners = labels.argmin(axis=1)
    counts = np.bincount(winners, minlength=labels.shape[1])
    return [1.0 / counts[winner] for winner in winners]


def batch_metrics(scores: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, float]:
    predicted = scores.argmax(dim=1)
    actual = labels.argmin(dim=1)
    rows = torch.arange(labels.size(0), device=labels.device)
    predicted_runtime = labels[rows, predicted]
    correct = int((predicted == actual).sum().item())
    solved = int((predicted_runtime < 2400).sum().item())
    par2 = float(predicted_runtime.sum().item())
    return correct, solved, par2


def run_epoch(
    *,
    model,
    loader,
    device,
    optimizer,
    scaler,
    margin: float,
    train: bool,
    amp: bool,
) -> dict:
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_solved = 0
    total_par2 = 0.0
    started = time.monotonic()

    for graphs, labels in loader:
        graphs = graphs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=amp):
                scores = model(
                    graphs.x,
                    graphs.edge_index,
                    graphs.edge_attr,
                    graphs.problemType,
                    graphs.batch,
                )
                loss = ranking_loss(scores, labels, margin)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        sample_count = labels.size(0)
        correct, solved, par2 = batch_metrics(scores.detach(), labels)
        total_loss += float(loss.detach().item()) * sample_count
        total_samples += sample_count
        total_correct += correct
        total_solved += solved
        total_par2 += par2

    torch.cuda.synchronize(device)
    duration = time.monotonic() - started
    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": total_correct / total_samples,
        "selected_solver_solved_rate": total_solved / total_samples,
        "selected_solver_mean_par2": total_par2 / total_samples,
        "samples": total_samples,
        "duration_seconds": duration,
        "samples_per_second": total_samples / duration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Sibyl GNN on a CUDA GPU")
    parser.add_argument("--data", required=True, type=Path, help="Graph cache root")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cross-valid", type=int, choices=range(10), default=0)
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "prefetch_factor"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.time_steps < 1:
        parser.error("--time-steps must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Sibyl training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with args.labels.open("r", encoding="utf-8") as label_file:
        labels = json.load(label_file)
    train_items = list(labels["train"].items())
    if not train_items:
        raise ValueError("Training labels are empty")
    solver_count = len(train_items[0][1])
    if solver_count < 2 or any(len(values) != solver_count for _, values in train_items):
        raise ValueError("Inconsistent solver label dimensions")

    fold_size = len(train_items) // 10
    val_start = fold_size * args.cross_valid
    val_end = fold_size * (args.cross_valid + 1)
    val_items = train_items[val_start:val_end]
    fit_items = train_items[:val_start] + train_items[val_end:]
    if not fit_items or not val_items:
        raise ValueError("Cross-validation split produced an empty dataset")

    train_dataset = LabeledGraphDataset(fit_items, args.data)
    val_dataset = LabeledGraphDataset(val_items, args.data)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        make_train_weights(fit_items),
        num_samples=len(fit_items),
        replacement=True,
        generator=generator,
    )
    options = loader_options(args)
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)

    model_config = {
        "passes": args.time_steps,
        "inputLayerSize": 67,
        "outputLayerSize": solver_count,
        "numAttentionLayers": 5,
        "mode": "cat",
        "pool": "attention",
        "k": 20,
        "dropout": args.dropout,
        "shouldJump": True,
    }
    model = GAT(**model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    amp = not args.no_amp
    scaler = GradScaler(enabled=amp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "amp": amp,
        "train_samples": len(fit_items),
        "validation_samples": len(val_items),
        "model_config": model_config,
        "epochs": [],
    }
    best_loss = math.inf
    training_started = time.monotonic()
    for epoch in range(args.epochs):
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            margin=args.margin,
            train=True,
            amp=amp,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            margin=args.margin,
            train=False,
            amp=amp,
        )
        scheduler.step(validation_metrics["loss"])
        epoch_report = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        report["epochs"].append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)

        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(
                {
                    "format": "smtgazer-sibyl-gat-v1",
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "training": {
                        "seed": args.seed,
                        "epoch": epoch,
                        "validation_loss": best_loss,
                    },
                },
                args.output,
            )

    report["best_validation_loss"] = best_loss
    report["duration_seconds"] = time.monotonic() - training_started
    report["checkpoint"] = str(args.output.resolve())
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] checkpoint={args.output}")
    print(f"[OK] training_metrics={args.metrics}")


if __name__ == "__main__":
    main()
