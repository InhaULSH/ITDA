"""
Train GNN baselines on a sampled YelpZip PyTorch Geometric graph.

Example:
    python TrainGNN.py --model sage --graph-path data/graph/graph_all_relations.pt
    python TrainGNN.py --model rgcn --class-weight --output-dir experiments/rgcn_all

This script trains only from the prepared graph Data object. It does not read
data/origin/yelpzip.csv or add raw user_id, prod_id, or date as model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH_PATH = PROJECT_DIR / "data" / "graph" / "graph_all_relations.pt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "experiments" / "manual_run"


def log(message: str) -> None:
    print(f"[TrainGNN] {message}")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def import_torch_and_pyg() -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install PyTorch first, then rerun this script. "
            "See: https://pytorch.org/get-started/locally/"
        ) from exc

    try:
        from torch_geometric.nn import GATConv, GCNConv, RGCNConv, SAGEConv
    except ImportError as exc:
        raise RuntimeError(
            "torch_geometric is not installed. Install PyTorch Geometric for your PyTorch/CUDA version, "
            "then rerun this script. See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        ) from exc

    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "GCNConv": GCNConv,
        "SAGEConv": SAGEConv,
        "GATConv": GATConv,
        "RGCNConv": RGCNConv,
    }


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def choose_device(torch: Any, device_arg: str) -> Any:
    if device_arg == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_graph(torch: Any, graph_path: Path, device: Any) -> Any:
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    log(f"Loading graph: {graph_path}")
    try:
        data = torch.load(graph_path, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(graph_path, map_location=device)

    required = ["x", "y", "edge_index", "train_mask", "valid_mask", "test_mask", "sampled_node_idx"]
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise ValueError(f"Graph Data object is missing required fields: {missing}")
    if data.x.ndim != 2:
        raise ValueError(f"data.x must be 2D, got {tuple(data.x.shape)}.")
    if data.y.ndim != 1 or data.y.shape[0] != data.x.shape[0]:
        raise ValueError(f"data.y must have shape [n_nodes], got {tuple(data.y.shape)}.")
    for name in ["train_mask", "valid_mask", "test_mask"]:
        mask = getattr(data, name)
        if mask.dtype != torch.bool:
            raise ValueError(f"data.{name} must be torch.bool, got {mask.dtype}.")
        if mask.shape != data.y.shape:
            raise ValueError(f"data.{name} shape must match y. mask={tuple(mask.shape)}, y={tuple(data.y.shape)}.")
    for name in ["target_mask", "train_target_mask", "valid_target_mask", "test_target_mask"]:
        if hasattr(data, name):
            mask = getattr(data, name)
            if mask.dtype != torch.bool:
                raise ValueError(f"data.{name} must be torch.bool, got {mask.dtype}.")
            if mask.shape != data.y.shape:
                raise ValueError(f"data.{name} shape must match y. mask={tuple(mask.shape)}, y={tuple(data.y.shape)}.")
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(f"data.edge_index must have shape [2, num_edges], got {tuple(data.edge_index.shape)}.")
    return data.to(device)


def build_mlp_class(nn: Any, F: Any) -> type:
    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1.")
            dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [2]
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
            self.dropout = dropout

        def forward(self, data: Any) -> Any:
            x = data.x
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.layers[-1](x)

    return MLP


def build_gnn_class(nn: Any, F: Any, conv_cls: Any) -> type:
    class ConvNet(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1.")
            if num_layers == 1:
                self.convs = nn.ModuleList([conv_cls(in_dim, 2)])
            else:
                self.convs = nn.ModuleList([conv_cls(in_dim, hidden_dim)])
                for _ in range(num_layers - 2):
                    self.convs.append(conv_cls(hidden_dim, hidden_dim))
                self.convs.append(conv_cls(hidden_dim, 2))
            self.dropout = dropout

        def forward(self, data: Any) -> Any:
            x = data.x
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, data.edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.convs[-1](x, data.edge_index)

    return ConvNet


def build_gat_class(nn: Any, F: Any, GATConv: Any, heads: int = 4) -> type:
    class GATNet(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1.")
            if num_layers == 1:
                self.convs = nn.ModuleList([GATConv(in_dim, 2, heads=1, dropout=dropout)])
            else:
                per_head = max(hidden_dim // heads, 1)
                self.convs = nn.ModuleList([GATConv(in_dim, per_head, heads=heads, dropout=dropout)])
                gat_hidden_dim = per_head * heads
                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(gat_hidden_dim, per_head, heads=heads, dropout=dropout))
                self.convs.append(GATConv(gat_hidden_dim, 2, heads=1, concat=False, dropout=dropout))
            self.dropout = dropout

        def forward(self, data: Any) -> Any:
            x = data.x
            for conv in self.convs[:-1]:
                x = F.elu(conv(x, data.edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.convs[-1](x, data.edge_index)

    return GATNet


def build_rgcn_class(nn: Any, F: Any, RGCNConv: Any, num_relations: int) -> type:
    class RGCNNet(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1.")
            if num_layers == 1:
                self.convs = nn.ModuleList([RGCNConv(in_dim, 2, num_relations=num_relations)])
            else:
                self.convs = nn.ModuleList([RGCNConv(in_dim, hidden_dim, num_relations=num_relations)])
                for _ in range(num_layers - 2):
                    self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
                self.convs.append(RGCNConv(hidden_dim, 2, num_relations=num_relations))
            self.dropout = dropout

        def forward(self, data: Any) -> Any:
            if not hasattr(data, "edge_type"):
                raise ValueError("R-GCN requires data.edge_type, but this graph does not contain edge_type.")
            x = data.x
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, data.edge_index, data.edge_type))
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.convs[-1](x, data.edge_index, data.edge_type)

    return RGCNNet


def make_sage_conv(SAGEConv: Any, in_dim: int, out_dim: int) -> Any:
    try:
        return SAGEConv(in_dim, out_dim, root_weight=False)
    except TypeError:
        return SAGEConv(in_dim, out_dim)


def build_relation_sage_class(
    torch: Any,
    nn: Any,
    F: Any,
    SAGEConv: Any,
    num_relations: int,
    relation_aggregation: str,
    relation_dropout: float,
) -> type:
    class RelationSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.self_linear = nn.Linear(in_dim, out_dim)
            self.rel_convs = nn.ModuleList(
                [make_sage_conv(SAGEConv, in_dim, out_dim) for _ in range(num_relations)]
            )
            if relation_aggregation == "learnable":
                self.relation_alpha = nn.Parameter(torch.zeros(num_relations))
            else:
                self.register_parameter("relation_alpha", None)

        def aggregate_relation_outputs(self, rel_outs: list[Any], active_relations: list[bool]) -> Any:
            stacked = torch.stack(rel_outs, dim=0)
            active = torch.tensor(active_relations, device=stacked.device, dtype=torch.bool)
            if int(active.sum().item()) == 0:
                return torch.zeros_like(stacked[0])

            if self.training and relation_dropout > 0:
                keep = (torch.rand(num_relations, device=stacked.device) >= relation_dropout) & active
                if int(keep.sum().item()) == 0:
                    active_indices = torch.flatnonzero(active)
                    keep[active_indices[torch.randint(len(active_indices), (1,), device=stacked.device)]] = True
                scale = 1.0 / max(1.0 - relation_dropout, 1e-12)
                stacked = stacked * keep.view(num_relations, 1, 1).to(stacked.dtype) * scale
                active = keep

            if relation_aggregation == "equal":
                return stacked[active].mean(dim=0)
            if relation_aggregation == "learnable":
                relation_logits = self.relation_alpha.masked_fill(~active, -torch.inf)
                weights = torch.softmax(relation_logits, dim=0)
                return torch.sum(stacked * weights.view(num_relations, 1, 1), dim=0)
            raise ValueError(f"Unsupported relation aggregation: {relation_aggregation}")

        def forward(self, x: Any, edge_index: Any, edge_type: Any) -> Any:
            self_out = self.self_linear(x)
            rel_outs = []
            active_relations = []
            for relation_id, conv in enumerate(self.rel_convs):
                relation_mask = edge_type == relation_id
                if int(relation_mask.sum().item()) > 0:
                    edge_index_r = edge_index[:, relation_mask]
                    rel_outs.append(conv(x, edge_index_r))
                    active_relations.append(True)
                else:
                    rel_outs.append(torch.zeros_like(self_out))
                    active_relations.append(False)
            return self_out + self.aggregate_relation_outputs(rel_outs, active_relations)

    class RelationSAGENet(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1.")
            if not 0 <= relation_dropout < 1:
                raise ValueError(f"--relation-dropout must be in [0, 1), got {relation_dropout}.")
            self.layers = nn.ModuleList()
            self.layers.append(RelationSAGELayer(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(RelationSAGELayer(hidden_dim, hidden_dim))
            self.classifier = nn.Linear(hidden_dim, 2)
            self.dropout = dropout

        def forward(self, data: Any) -> Any:
            if not hasattr(data, "edge_type"):
                raise ValueError("relation_sage requires data.edge_type, but this graph does not contain edge_type.")
            x = data.x
            for layer in self.layers:
                x = F.relu(layer(x, data.edge_index, data.edge_type))
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.classifier(x)

        def relation_weights(self) -> dict[str, list[float]]:
            if relation_aggregation != "learnable":
                return {}
            weights: dict[str, list[float]] = {}
            for layer_idx, layer in enumerate(self.layers):
                weight = torch.softmax(layer.relation_alpha.detach().cpu(), dim=0).numpy()
                weights[f"layer_{layer_idx}"] = [float(value) for value in weight]
            return weights

    return RelationSAGENet


def build_relation_sage_mlp_class(
    torch: Any,
    nn: Any,
    F: Any,
    SAGEConv: Any,
    num_relations: int,
    relation_aggregation: str,
    relation_dropout: float,
) -> type:
    class RelationSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.self_linear = nn.Linear(in_dim, out_dim)
            self.rel_convs = nn.ModuleList(
                [make_sage_conv(SAGEConv, in_dim, out_dim) for _ in range(num_relations)]
            )
            if relation_aggregation == "learnable":
                self.relation_alpha = nn.Parameter(torch.zeros(num_relations))
            else:
                self.register_parameter("relation_alpha", None)

        def aggregate_relation_outputs(self, rel_outs: list[Any], active_relations: list[bool]) -> Any:
            stacked = torch.stack(rel_outs, dim=0)
            active = torch.tensor(active_relations, device=stacked.device, dtype=torch.bool)
            if int(active.sum().item()) == 0:
                return torch.zeros_like(stacked[0])

            if self.training and relation_dropout > 0:
                keep = (torch.rand(num_relations, device=stacked.device) >= relation_dropout) & active
                if int(keep.sum().item()) == 0:
                    active_indices = torch.flatnonzero(active)
                    keep[active_indices[torch.randint(len(active_indices), (1,), device=stacked.device)]] = True
                scale = 1.0 / max(1.0 - relation_dropout, 1e-12)
                stacked = stacked * keep.view(num_relations, 1, 1).to(stacked.dtype) * scale
                active = keep

            if relation_aggregation == "equal":
                return stacked[active].mean(dim=0)
            if relation_aggregation == "learnable":
                relation_logits = self.relation_alpha.masked_fill(~active, -torch.inf)
                weights = torch.softmax(relation_logits, dim=0)
                return torch.sum(stacked * weights.view(num_relations, 1, 1), dim=0)
            raise ValueError(f"Unsupported relation aggregation: {relation_aggregation}")

        def forward(self, x: Any, edge_index: Any, edge_type: Any) -> Any:
            self_out = self.self_linear(x)
            rel_outs = []
            active_relations = []
            for relation_id, conv in enumerate(self.rel_convs):
                relation_mask = edge_type == relation_id
                if int(relation_mask.sum().item()) > 0:
                    edge_index_r = edge_index[:, relation_mask]
                    rel_outs.append(conv(x, edge_index_r))
                    active_relations.append(True)
                else:
                    rel_outs.append(torch.zeros_like(self_out))
                    active_relations.append(False)
            return self_out + self.aggregate_relation_outputs(rel_outs, active_relations)

    class RelationSageMlpNet(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be at least 1.")
            if not 0 <= relation_dropout < 1:
                raise ValueError(f"--relation-dropout must be in [0, 1), got {relation_dropout}.")

            self.self_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.graph_layers = nn.ModuleList()
            self.graph_layers.append(RelationSAGELayer(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.graph_layers.append(RelationSAGELayer(hidden_dim, hidden_dim))
            self.classifier = nn.Linear(hidden_dim * 2, 2)
            self.dropout = dropout

        def forward(self, data: Any) -> Any:
            if not hasattr(data, "edge_type"):
                raise ValueError("relation_sage_mlp requires data.edge_type, but this graph does not contain edge_type.")
            self_hidden = self.self_mlp(data.x)
            graph_hidden = data.x
            for layer in self.graph_layers:
                graph_hidden = F.relu(layer(graph_hidden, data.edge_index, data.edge_type))
                graph_hidden = F.dropout(graph_hidden, p=self.dropout, training=self.training)
            return self.classifier(torch.cat([self_hidden, graph_hidden], dim=1))

        def relation_weights(self) -> dict[str, list[float]]:
            if relation_aggregation != "learnable":
                return {}
            weights: dict[str, list[float]] = {}
            for layer_idx, layer in enumerate(self.graph_layers):
                weight = torch.softmax(layer.relation_alpha.detach().cpu(), dim=0).numpy()
                weights[f"layer_{layer_idx}"] = [float(value) for value in weight]
            return weights

    return RelationSageMlpNet


def create_model(imports: dict[str, Any], data: Any, args: argparse.Namespace) -> Any:
    nn = imports["nn"]
    F = imports["F"]
    torch = imports["torch"]
    in_dim = int(data.x.shape[1])

    if args.model == "mlp":
        model_cls = build_mlp_class(nn, F)
    elif args.model == "gcn":
        model_cls = build_gnn_class(nn, F, imports["GCNConv"])
    elif args.model == "sage":
        model_cls = build_gnn_class(nn, F, imports["SAGEConv"])
    elif args.model == "relation_sage":
        if not hasattr(data, "edge_type"):
            raise ValueError("relation_sage requires data.edge_type, but this graph does not contain edge_type.")
        if data.edge_type.numel() == 0:
            raise ValueError("relation_sage requires non-empty edge_type.")
        num_relations = max(int(data.edge_type.max().item()) + 1, 3)
        model_cls = build_relation_sage_class(
            torch,
            nn,
            F,
            imports["SAGEConv"],
            num_relations,
            args.relation_aggregation,
            args.relation_dropout,
        )
    elif args.model == "relation_sage_mlp":
        if not hasattr(data, "edge_type"):
            raise ValueError("relation_sage_mlp requires data.edge_type, but this graph does not contain edge_type.")
        if data.edge_type.numel() == 0:
            raise ValueError("relation_sage_mlp requires non-empty edge_type.")
        num_relations = max(int(data.edge_type.max().item()) + 1, 3)
        model_cls = build_relation_sage_mlp_class(
            torch,
            nn,
            F,
            imports["SAGEConv"],
            num_relations,
            args.relation_aggregation,
            args.relation_dropout,
        )
    elif args.model == "gat":
        model_cls = build_gat_class(nn, F, imports["GATConv"], heads=4)
    elif args.model == "rgcn":
        if not hasattr(data, "edge_type"):
            raise ValueError("R-GCN requires data.edge_type, but this graph does not contain edge_type.")
        if data.edge_type.numel() == 0:
            raise ValueError("R-GCN requires non-empty edge_type.")
        num_relations = int(data.edge_type.max().item()) + 1
        model_cls = build_rgcn_class(nn, F, imports["RGCNConv"], num_relations=num_relations)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model_cls(in_dim, args.hidden_dim, args.num_layers, args.dropout)


def class_weights(torch: Any, y: Any, train_mask: Any, device: Any) -> Any:
    train_y = y[train_mask]
    counts = torch.bincount(train_y, minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError(f"Cannot compute class weights because a train class is missing. counts={counts.tolist()}")
    weights = counts.sum() / (2.0 * counts)
    return weights.to(device)


def class_balanced_weights(torch: Any, y: Any, train_mask: Any, beta: float, device: Any) -> Any:
    if not 0 <= beta < 1:
        raise ValueError(f"--class-balanced-beta must be in [0, 1), got {beta}.")
    train_y = y[train_mask]
    counts = torch.bincount(train_y, minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError(f"Cannot compute class-balanced weights because a train class is missing. counts={counts.tolist()}")
    effective_num = 1.0 - torch.pow(torch.full_like(counts, float(beta)), counts)
    weights = (1.0 - float(beta)) / torch.clamp(effective_num, min=1e-12)
    weights = weights / weights.mean()
    return weights.to(device)


def select_eval_masks(data: Any, mask_mode: str) -> dict[str, Any]:
    if mask_mode == "target":
        required = ["train_target_mask", "valid_target_mask", "test_target_mask"]
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(f"--mask-mode target requires graph fields: {missing}")
        masks = {
            "train": data.train_target_mask,
            "valid": data.valid_target_mask,
            "test": data.test_target_mask,
        }
    elif mask_mode == "split":
        masks = {
            "train": data.train_mask,
            "valid": data.valid_mask,
            "test": data.test_mask,
        }
    else:
        raise ValueError(f"Unsupported mask mode: {mask_mode}")

    for name, mask in masks.items():
        if int(mask.sum().item()) == 0:
            raise ValueError(f"{mask_mode} {name} mask has no nodes.")
    return masks


def mask_summary(data: Any, masks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        y = data.y[mask]
        fake = int((y == 1).sum().item())
        real = int((y == 0).sum().item())
        total = fake + real
        summary[name] = {
            "nodes": total,
            "fake": fake,
            "real": real,
            "fake_rate": float(fake / total) if total else None,
        }
    return summary


def relation_edge_counts(data: Any) -> dict[str, int]:
    if not hasattr(data, "edge_type") or data.edge_type.numel() == 0:
        return {}
    edge_type = tensor_to_numpy(data.edge_type).astype(np.int64)
    values, counts = np.unique(edge_type, return_counts=True)
    summary = {str(int(value)): int(count) for value, count in zip(values, counts)}
    for relation_id in range(3):
        summary.setdefault(str(relation_id), 0)
    return dict(sorted(summary.items(), key=lambda item: int(item[0])))


def learnable_relation_weights(model: Any) -> dict[str, list[float]] | None:
    if hasattr(model, "relation_weights"):
        weights = model.relation_weights()
        return weights if weights else None
    return None


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def safe_metrics(y_true: np.ndarray, prob_fake: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (prob_fake >= threshold).astype(np.int64)
    if len(np.unique(y_true)) < 2:
        pr_auc = None
        roc_auc = None
    else:
        pr_auc = float(average_precision_score(y_true, prob_fake))
        roc_auc = float(roc_auc_score(y_true, prob_fake))
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def find_best_threshold(
    y_true: np.ndarray,
    prob_fake: np.ndarray,
    strategy: str = "macro_f1",
) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = -1.0
    max_predicted_positive_rate = None
    if strategy == "prevalence_constrained_macro_f1":
        max_predicted_positive_rate = float(np.mean(y_true))
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (prob_fake >= threshold).astype(np.int64)
        if max_predicted_positive_rate is not None and float(np.mean(pred)) > max_predicted_positive_rate:
            continue
        score = float(f1_score(y_true, pred, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate(
    imports: dict[str, Any],
    model: Any,
    data: Any,
    mask: Any,
    threshold: float | None = None,
    threshold_strategy: str = "macro_f1",
) -> dict[str, Any]:
    torch = imports["torch"]
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)[:, 1]
    y_true = tensor_to_numpy(data.y[mask]).astype(np.int64)
    prob_fake = tensor_to_numpy(probs[mask]).astype(np.float64)
    if threshold is None:
        threshold, best_macro_f1 = find_best_threshold(y_true, prob_fake, strategy=threshold_strategy)
    else:
        best_macro_f1 = None
    metrics = safe_metrics(y_true, prob_fake, threshold)
    metrics["threshold"] = float(threshold)
    if best_macro_f1 is not None:
        metrics["best_threshold_macro_f1"] = float(best_macro_f1)
    return metrics


def save_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "train_loss", "valid_pr_auc", "valid_macro_f1", "valid_threshold"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path: Path, data: Any, prob_fake: np.ndarray, threshold: float) -> pd.DataFrame:
    sampled_idx = tensor_to_numpy(data.sampled_node_idx).astype(np.int64)
    y_true = tensor_to_numpy(data.y).astype(np.int64)
    pred_label = (prob_fake >= threshold).astype(np.int64)
    split = np.full(len(y_true), "unknown", dtype=object)
    split[tensor_to_numpy(data.train_mask).astype(bool)] = "train"
    split[tensor_to_numpy(data.valid_mask).astype(bool)] = "valid"
    split[tensor_to_numpy(data.test_mask).astype(bool)] = "test"
    is_target_node = (
        tensor_to_numpy(data.target_mask).astype(bool)
        if hasattr(data, "target_mask")
        else np.ones(len(y_true), dtype=bool)
    )

    df = pd.DataFrame(
        {
            "sampled_node_idx": sampled_idx,
            "y_true": y_true,
            "prob_fake": prob_fake,
            "pred_label": pred_label,
            "split": split,
            "is_target_node": is_target_node,
        }
    )
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def train(args: argparse.Namespace) -> None:
    imports = import_torch_and_pyg()
    torch = imports["torch"]
    nn = imports["nn"]
    set_seed(torch, args.seed)
    device = choose_device(torch, args.device)
    log(f"Using device: {device}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_graph(torch, args.graph_path, device)
    eval_masks = select_eval_masks(data, args.mask_mode)
    model = create_model(imports, data, args).to(device)

    if args.class_balanced_loss:
        weight = class_balanced_weights(torch, data.y, eval_masks["train"], args.class_balanced_beta, device)
    elif args.class_weight:
        weight = class_weights(torch, data.y, eval_masks["train"], device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["graph_path"] = path_for_summary(args.graph_path)
    config["output_dir"] = path_for_summary(output_dir)
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, ensure_ascii=False, indent=2)

    best_metric = -float("inf")
    best_epoch = 0
    best_threshold = 0.5
    best_state = None
    epochs_without_improvement = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[eval_masks["train"]], data.y[eval_masks["train"]])
        loss.backward()
        optimizer.step()

        valid_metrics = evaluate(
            imports,
            model,
            data,
            eval_masks["valid"],
            threshold=None,
            threshold_strategy=args.threshold_strategy,
        )
        valid_pr_auc = valid_metrics["pr_auc"]
        valid_macro_f1 = valid_metrics["macro_f1"]
        valid_threshold = valid_metrics["threshold"]
        current_metric = valid_macro_f1 if args.early_stop_metric == "valid_macro_f1" else valid_pr_auc
        current_metric = -float("inf") if current_metric is None else float(current_metric)

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "valid_pr_auc": valid_pr_auc,
                "valid_macro_f1": valid_macro_f1,
                "valid_threshold": valid_threshold,
            }
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_threshold = float(valid_threshold)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            log(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"valid_pr_auc={valid_pr_auc if valid_pr_auc is not None else 'NA'} "
                f"valid_macro_f1={valid_macro_f1:.4f}"
            )

        if epochs_without_improvement >= args.patience:
            log(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    relation_counts = relation_edge_counts(data)
    learned_relation_weights = learnable_relation_weights(model)
    torch.save(
        {
            "model_state_dict": best_state,
            "model": args.model,
            "in_dim": int(data.x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "relation_aggregation": args.relation_aggregation
            if args.model in {"relation_sage", "relation_sage_mlp"}
            else None,
            "relation_dropout": float(args.relation_dropout)
            if args.model in {"relation_sage", "relation_sage_mlp"}
            else None,
            "relation_edge_counts": relation_counts if args.model in {"relation_sage", "relation_sage_mlp"} else None,
            "learnable_relation_weights": learned_relation_weights,
            "best_epoch": int(best_epoch),
            "best_threshold": float(best_threshold),
        },
        output_dir / "best_model.pt",
    )

    valid_metrics = evaluate(imports, model, data, eval_masks["valid"], threshold=best_threshold)
    test_metrics = evaluate(imports, model, data, eval_masks["test"], threshold=best_threshold)
    train_metrics = evaluate(imports, model, data, eval_masks["train"], threshold=best_threshold)

    model.eval()
    with torch.no_grad():
        logits = model(data)
        prob_fake = tensor_to_numpy(torch.softmax(logits, dim=1)[:, 1]).astype(np.float64)

    predictions_all = save_predictions(output_dir / "predictions_all.csv", data, prob_fake, best_threshold)
    prediction_test = predictions_all.loc[predictions_all["split"] == "test"].copy()
    if args.mask_mode == "target":
        prediction_test = prediction_test.loc[prediction_test["is_target_node"]].copy()
    prediction_test.to_csv(
        output_dir / "prediction_test.csv",
        index=False,
        encoding="utf-8",
    )
    save_training_log(output_dir / "training_log.csv", log_rows)

    metrics = {
        "model": args.model,
        "best_epoch": int(best_epoch),
        "best_valid_metric": float(best_metric),
        "early_stop_metric": args.early_stop_metric,
        "threshold_selection": args.threshold_strategy,
        "mask_mode": args.mask_mode,
        "mask_summary": mask_summary(data, eval_masks),
        "loss_weighting": "class_balanced" if args.class_balanced_loss else ("inverse_frequency" if args.class_weight else "none"),
        "class_balanced_beta": float(args.class_balanced_beta) if args.class_balanced_loss else None,
        "best_threshold": float(best_threshold),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    if args.model in {"relation_sage", "relation_sage_mlp"}:
        metrics["relation_aggregation"] = args.relation_aggregation
        metrics["relation_dropout"] = float(args.relation_dropout)
        metrics["relation_edge_counts"] = relation_counts
        if learned_relation_weights is not None:
            metrics["learnable_relation_weights"] = learned_relation_weights
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, ensure_ascii=False, indent=2)

    log(f"Saved outputs to: {output_dir}")
    log(f"Best epoch={best_epoch}, best_threshold={best_threshold:.2f}, test_macro_f1={test_metrics['macro_f1']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP/GNN models on a prepared sampled PyG graph.")
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH, help="Path to PyG graph .pt file")
    parser.add_argument(
        "--model",
        choices=["mlp", "gcn", "sage", "gat", "rgcn", "relation_sage", "relation_sage_mlp"],
        required=True,
        help="Model type",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Experiment output directory")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device: auto, xpu, cpu, cuda, cuda:0, ...")
    parser.add_argument("--class-weight", action="store_true", help="Use train-label class weights")
    parser.add_argument(
        "--class-balanced-loss",
        action="store_true",
        help="Use class-balanced effective-number weights instead of inverse-frequency class weights.",
    )
    parser.add_argument(
        "--class-balanced-beta",
        type=float,
        default=0.999,
        help="Beta for class-balanced effective-number weighting.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["split", "target"],
        default="split",
        help=(
            "Which nodes are used for loss/validation/test metrics. "
            "target uses seed nodes only when the graph contains target masks; context nodes still pass messages."
        ),
    )
    parser.add_argument(
        "--relation-aggregation",
        choices=["equal", "learnable"],
        default="equal",
        help="How relation_sage combines relation-wise SAGE outputs",
    )
    parser.add_argument(
        "--relation-dropout",
        type=float,
        default=0.0,
        help="Drop complete relation message outputs during relation_sage training",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["valid_pr_auc", "valid_macro_f1"],
        default="valid_pr_auc",
        help="Validation metric used for best-model selection and early stopping",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["macro_f1", "pr_auc"],
        default="macro_f1",
        help="Deprecated compatibility option; threshold selection always uses valid Macro F1",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["macro_f1", "prevalence_constrained_macro_f1"],
        default="macro_f1",
        help=(
            "How to choose the decision threshold on validation data. "
            "prevalence_constrained_macro_f1 maximizes Macro F1 while keeping the predicted "
            "positive rate no higher than the observed validation positive rate."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception as exc:
        print(f"[TrainGNN][ERROR] {exc}", file=sys.stderr)
        raise
