#!/usr/bin/env python3
"""Render the selected nominal-pose transmitter/receiver WPT geometry."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
WIRE_WIDTH = .68
OUT = ROOT/"essay"/"figures"/"fig06_wpt_selected_geometry"


def draw(ax, ns, nt, spacing, inner_radius, z, color):
    """Draw concentric regular polygons using the benchmark's mm geometry."""
    angles = 2*np.pi*np.arange(ns)/ns
    unit_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    for turn in range(nt):
        radius = inner_radius+turn*(spacing+WIRE_WIDTH)
        vertices = radius*unit_vertices
        xy = np.vstack((vertices, vertices[0]))
        ax.plot(xy[:, 0], xy[:, 1], np.full(len(xy), z), color=color,
                lw=.85, alpha=.9)


def main():
    result = json.loads((ROOT/"essay"/"data"/"wpt_8d_automated_cei"/"result.json").read_text())
    ns_t, nt_t, s_t, ri_t, ns_r, nt_r, s_r, ri_r = result["design"]
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig = plt.figure(figsize=(3.5, 2.55))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, round(ns_t), round(nt_t), s_t, ri_t, 0, "#d95f02")
    draw(ax, round(ns_r), round(nt_r), s_r, ri_r, 70, "#1b9e77")
    ax.plot([0, 0], [0, 0], [0, 70], color="0.35", ls="--", lw=1)
    ax.text(3, 3, 35, "$70$ mm", fontsize=7)
    ax.set(xlabel="$x$ (mm)", ylabel="$y$ (mm)", zlabel="$z$ (mm)")
    ax.set_xticks([-60, 0, 60])
    ax.set_yticks([-60, 0, 60])
    ax.set_zticks([0, 70])
    ax.legend(handles=[
        Line2D([0], [0], color="#d95f02", lw=1.4,
               label="Tx: 9 sides, 7 turns"),
        Line2D([0], [0], color="#1b9e77", lw=1.4,
               label="Rx: 7 sides, 7 turns"),
    ], loc="upper left", frameon=True, facecolor="white", edgecolor="0.8",
       fontsize=6.3, borderpad=.3, handlelength=1.5)
    ax.set_box_aspect((1, 1, .62))
    ax.view_init(elev=23, azim=-58)
    ax.grid(False)
    fig.tight_layout(pad=.25)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix("."+suffix), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
