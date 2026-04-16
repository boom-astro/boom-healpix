#!/usr/bin/env python3
"""
Plot benchmark results from JSON output files.

Generates comparison plots for:
  1. Query time vs cone radius (per method)
  2. Query time vs catalog size (scaling)
  3. Index build time and size comparison

Usage:
  # Single run
  cargo run --release --bin benchmark -- --catalog-size 1000000 --output-json results_1M.json
  python plot.py results_1M.json

  # Scaling comparison (multiple catalog sizes)
  for n in 100000 500000 1000000 2000000; do
    cargo run --release --bin benchmark -- --catalog-size $n --output-json results_${n}.json
  done
  python plot.py results_*.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = {
    "2dsphere": {"color": "#1f77b4", "marker": "o", "label": "2dsphere"},
    "hpx_ranges": {"color": "#ff7f0e", "marker": "s", "label": "HEALPix ranges"},
    "hpx_in": {"color": "#2ca02c", "marker": "^", "label": "HEALPix $in"},
}

# Consistent ordering for bar charts (index build time, index size)
INDEX_ORDER = [
    ("2dsphere", "2dsphere", "#1f77b4"),
    ("hpx_29", "hpx (depth 29)", "#ff7f0e"),
    ("hpx_16", "hpx (depth 16)", "#2ca02c"),
]


def load_results(paths):
    """Load one or more JSON result files."""
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def plot_single_run(data, output_path="benchmark.png"):
    """Plot results from a single benchmark run."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Query time vs radius ---
    ax = axes[0]
    for method, style in METHODS.items():
        qs = [q for q in data["queries"] if q["method"] == method]
        radii = [q["radius_arcsec"] for q in qs]
        times = [q["time_secs"] / data["n_queries"] * 1000 for q in qs]  # ms per query
        ax.plot(radii, times, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Cone radius (arcsec)")
    ax.set_ylabel("Time per query (ms)")
    ax.set_title(f"Query latency ({data['catalog_size']:,} docs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Index build time ---
    ax = axes[1]
    build = data["index_build_secs"]
    # Use consistent order, skip missing keys
    bar_items = [(label, build.get(key, 0), color)
                 for key, label, color in INDEX_ORDER if key in build]
    names = [b[0] for b in bar_items]
    times = [b[1] for b in bar_items]
    colors_bar = [b[2] for b in bar_items]
    bars = ax.bar(names, times, color=colors_bar)
    ax.set_ylabel("Build time (s)")
    ax.set_title("Index build time")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=10)

    # --- Panel 3: Index size ---
    ax = axes[2]
    sizes = data["index_size_bytes"]
    bar_items = [(label, sizes.get(key, 0) / 1024 / 1024, color)
                 for key, label, color in INDEX_ORDER if key in sizes]
    names = [b[0] for b in bar_items]
    mb = [b[1] for b in bar_items]
    colors_bar = [b[2] for b in bar_items]
    bars = ax.bar(names, mb, color=colors_bar)
    ax.set_ylabel("Index size (MB)")
    ax.set_title("Index size on disk")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, mb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{m:.1f} MB", ha="center", va="bottom", fontsize=10)

    fig.suptitle("HEALPix vs 2dsphere — MongoDB Cone Search Benchmark", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


def plot_scaling(all_data, output_path="scaling.png"):
    """Plot scaling across multiple catalog sizes."""
    # Sort by catalog size
    all_data.sort(key=lambda d: d["catalog_size"])

    # Pick a representative radius (10" — typical cross-match)
    target_radius = 10.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Query time vs catalog size ---
    ax = axes[0]
    for method, style in METHODS.items():
        sizes = []
        times = []
        for data in all_data:
            qs = [q for q in data["queries"]
                  if q["method"] == method and q["radius_arcsec"] == target_radius]
            if qs:
                sizes.append(data["catalog_size"])
                times.append(qs[0]["time_secs"] / data["n_queries"] * 1000)
        if sizes:
            ax.plot(sizes, times, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Catalog size (documents)")
    ax.set_ylabel("Time per query (ms)")
    ax.set_title(f'Query scaling at {target_radius}" radius')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Index build time vs catalog size ---
    ax = axes[1]
    for key, label, color in INDEX_ORDER:
        cat_sizes = []
        times = []
        for data in all_data:
            if key in data["index_build_secs"]:
                cat_sizes.append(data["catalog_size"])
                times.append(data["index_build_secs"][key])
        if cat_sizes:
            ax.plot(cat_sizes, times, marker="o", color=color,
                    label=label, linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Catalog size (documents)")
    ax.set_ylabel("Build time (s)")
    ax.set_title("Index build time scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Index size vs catalog size ---
    ax = axes[2]
    for key, label, color in INDEX_ORDER:
        cat_sizes = []
        idx_mb = []
        for data in all_data:
            if key in data["index_size_bytes"]:
                cat_sizes.append(data["catalog_size"])
                idx_mb.append(data["index_size_bytes"][key] / 1024 / 1024)
        if cat_sizes:
            ax.plot(cat_sizes, idx_mb, marker="o", color=color,
                    label=label, linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Catalog size (documents)")
    ax.set_ylabel("Index size (MB)")
    ax.set_title("Index size scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("HEALPix vs 2dsphere — Scaling with Catalog Size", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py results.json [results2.json ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    all_data = load_results(paths)

    if len(all_data) == 1:
        plot_single_run(all_data[0])
    else:
        # Both single-run plots for each, plus scaling comparison
        for data in all_data:
            n = data["catalog_size"]
            plot_single_run(data, output_path=f"benchmark_{n}.png")
        plot_scaling(all_data)
