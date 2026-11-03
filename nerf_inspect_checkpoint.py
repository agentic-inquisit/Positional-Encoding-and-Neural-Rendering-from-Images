"""Read a .pt checkpoint, print its layer shapes, and save a visualization of its weights.

Usage:
    python nerf_inspect_checkpoint.py nerf_model_nerf.pt [more.pt ...]
    python nerf_inspect_checkpoint.py  # defaults to every nerf_model_*.pt in this directory

For each checkpoint, saves "<checkpoint_name>_weights.png": one heatmap per
2D weight matrix (sorted by layer) plus a histogram of every bias vector,
so you can see each layer's shape and the spread of its learned values at
a glance.
"""

import argparse
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def inspect(path, out_dir):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    state_dict = torch.load(path, map_location="cpu")

    print(f"\n{path}  ({size_mb:.2f} MB)")
    total_params = 0
    weights, biases = [], []
    for name, tensor in state_dict.items():
        n = tensor.numel()
        total_params += n
        print(f"  {name:<24} {tuple(tensor.shape)!s:<20} {tensor.dtype}")
        if tensor.dim() == 2:
            weights.append((name, tensor))
        elif tensor.dim() == 1:
            biases.append((name, tensor))
    print(f"  total parameters: {total_params:,}")

    n_plots = len(weights) + (1 if biases else 0)
    n_cols = min(4, max(1, n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, (name, tensor) in zip(axes, weights):
        arr = tensor.numpy()
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-arr.std() * 3, vmax=arr.std() * 3)
        ax.set_title(f"{name}\n{tuple(tensor.shape)}", fontsize=9)
        ax.set_xlabel("in"); ax.set_ylabel("out")
        fig.colorbar(im, ax=ax, fraction=0.046)

    if biases:
        ax = axes[len(weights)]
        for name, tensor in biases:
            ax.hist(tensor.numpy(), bins=40, alpha=0.6, label=name.split(".")[-2] if "." in name else name)
        ax.set_title("bias distributions", fontsize=9)
        ax.legend(fontsize=6)

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(f"{os.path.basename(path)} — {total_params:,} params")
    fig.tight_layout()
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_weights.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved visualization -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Path(s) to .pt checkpoint files")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    paths = args.paths or sorted(glob.glob(os.path.join(here, "nerf_model_*.pt")))
    if not paths:
        print("No .pt files given and none found as nerf_model_*.pt", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        inspect(path, out_dir=here)


if __name__ == "__main__":
    main()
