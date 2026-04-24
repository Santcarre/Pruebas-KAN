"""
Visualización ligera de aristas/splines KAN tras cachear activaciones.

Compatible con PyKAN >= 0.2.x (API `get_act`).

Uso correcto:
    # 1. Ejecutar forward
    out = model(x)
    # 2. Cachear activaciones con get_act
    model.get_act(x)
    # 3. Llamar a las funciones de este módulo
    plot_kan_edges_light(model, layer=0)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def estimate_kan_plot_load(model) -> list[dict[str, int]]:
    rows = []
    for layer in range(len(model.width_in) - 1):
        in_dim = int(model.width_in[layer])
        out_dim = int(model.width_out[layer + 1])
        rows.append(
            {
                "layer": layer,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "edges": in_dim * out_dim,
            }
        )
    return rows


def _cached_edge_pairs(model, layer: int, top_k: int, metric: str = "mean_abs") -> list[tuple[int, int]]:
    if getattr(model, "spline_postacts", None) is None:
        raise RuntimeError(
            "No cached spline activations found. "
            "Run model.get_act(x) after a forward pass. "
            "(PyKAN >= 0.2.x: save_act=True ya no existe en forward())"
        )

    spline_postacts = model.spline_postacts[layer]
    if metric == "mean_abs":
        scores = spline_postacts.abs().mean(dim=0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    flat_idx = torch.argsort(scores.flatten(), descending=True)[:top_k]
    out_dim, in_dim = scores.shape
    pairs = []
    for idx in flat_idx.tolist():
        out_idx = idx // in_dim
        in_idx = idx % in_dim
        pairs.append((in_idx, out_idx))
    return pairs


def plot_kan_edges_light(
    model,
    layer: int = 0,
    top_k: int = 12,
    metric: str = "mean_abs",
    sample_points: bool = False,
    max_points: int | None = 256,
    sort_x: bool = True,
    cols: int = 4,
    figsize_scale: float = 1.0,
):
    if getattr(model, "acts", None) is None or getattr(model, "spline_postacts", None) is None:
        raise RuntimeError(
            "KAN activations are not cached. "
            "Run model.get_act(x) after a forward pass. "
            "(PyKAN >= 0.2.x: save_act=True ya no existe en forward())"
        )

    pairs = _cached_edge_pairs(model, layer=layer, top_k=top_k, metric=metric)
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols * figsize_scale, 3 * rows * figsize_scale))
    axes = np.array(axes).reshape(-1)

    acts = model.acts[layer]
    spline_postacts = model.spline_postacts[layer]

    for ax, (in_idx, out_idx) in zip(axes, pairs):
        x = acts[:, in_idx].detach().cpu().numpy()
        y = spline_postacts[:, out_idx, in_idx].detach().cpu().numpy()

        if sort_x:
            order = np.argsort(x)
            x = x[order]
            y = y[order]

        if max_points is not None and len(x) > max_points:
            sel = np.linspace(0, len(x) - 1, max_points, dtype=int)
            x = x[sel]
            y = y[sel]

        ax.plot(x, y, lw=2)
        if sample_points:
            ax.scatter(x, y, s=12, alpha=0.7)
        ax.set_title(f"layer {layer} | in {in_idx} -> out {out_idx}")
        ax.grid(alpha=0.2)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.tight_layout()
    return fig, pairs


def save_kan_edges_light(
    model,
    output_path: str | Path,
    **kwargs,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, pairs = plot_kan_edges_light(model, **kwargs)
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return output_path, pairs
