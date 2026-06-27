"""Regenerate Fig 2 with a UNIFIED pooled chr17+chr22 intron baseline.

Replaces the panel-(b) ``vs chr22 background`` d values with values computed
locally by ``compute_fig2_unified_d.py`` against the same pooled intron
baseline used by panel (a). This collapses the prior baseline mismatch
flagged by reviewer #4 (panel-a splice ``d ≈ -0.36`` vs panel-b splice
``d = -0.433``).

Sources for the new d's:
  splice donor (pooled chr17+chr22) :  -0.354   (fig_v9_meta panel_a_sig: -0.358)
  ENCODE cCRE-ELS chr22 (v4)        :  -0.118
  GTEx Whole-Blood eQTL chr22       :  +0.064
  GWAS Catalog chr22 (latest)       :  +0.021
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

V3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_DIR.parent
META = REPO_ROOT / "results" / "figures_v3" / "fig_v9_meta.json"
OUT_DIR = V3_DIR / "figures"

BLUE = "#1f77b4"
RED = "#d62728"
ORANGE = "#ff7f0e"
GREY = "#bcbcbc"
LIGHT_GREY = "#dcdcdc"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sig_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def fig_2_shallowness_unified() -> None:
    meta = json.loads(META.read_text())
    intron_baseline = float(meta["shallowness"]["intron_baseline"])

    # Panel (a) per-context mean settling depth (pooled chr17+chr22).
    # Splice donor / acceptor values reproduced from panel_a_sig in
    # fig_v9_meta.json (panel-a d already used this baseline).
    panel_a = [
        ("splice donor",      25.55, -0.36, 1e-30, BLUE,       False),
        ("splice acceptor",   25.96, -0.34, 1e-30, BLUE,       False),
        ("intron (baseline)", 27.72,  0.00, 1.00,  GREY,       False),
        ("3$'$ UTR",          27.74, -0.02, 1e-63, LIGHT_GREY, False),
        ("coding exon",       28.40, +0.08, 1e-105, LIGHT_GREY, False),
        ("intergenic",        28.66, +0.16, 1e-30, LIGHT_GREY, False),
        ("5$'$ UTR$^*$",      29.22, +0.20, 1e-30, LIGHT_GREY, True),
    ]

    # Panel (b): UNIFIED baseline — values from compute_fig2_unified_d.py.
    # (label, d, p_for_stars, color, n)
    panel_b_unified = [
        ("splice donor\n(reference)",  -0.354, 0.0,         BLUE,    340_621),
        ("ENCODE cCRE-ELS",            -0.118, 0.0,         RED,     9_044_492),
        ("GTEx Whole-Blood eQTL",      +0.064, 1e-30,       ORANGE,  32_456),
        ("GWAS Catalog chr22",         +0.021, 5e-2,        ORANGE,  6_898),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.35),
                             gridspec_kw={"width_ratios": [1.12, 1.0]})

    # ------------------------------------------------------------------
    # Panel (a) — identical to existing
    # ------------------------------------------------------------------
    ax = axes[0]
    n = len(panel_a)
    y = list(range(n))
    means = [r[1] for r in panel_a]
    colors = [r[4] for r in panel_a]
    ax.barh(y, means, color=colors, edgecolor="black", lw=0.45, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in panel_a])
    ax.invert_yaxis()
    ax.axvline(intron_baseline, color=RED, ls="--", lw=1.0, alpha=0.75)
    for yi, (lab, mean_c, d, p, _col, _ast) in enumerate(panel_a):
        if lab.startswith("intron"):
            text = f"{mean_c:.2f}"
        else:
            text = f"{mean_c:.2f}   $d={d:+.2f}$"
        ax.text(mean_c + 0.05, yi, text, va="center", fontsize=8)
    ax.set_xlim(24.0, 30.5)
    ax.set_xlabel("mean settling depth $\\bar c$ (pooled chr17+chr22)")
    ax.set_title("(a) Splice contexts settle earlier than introns",
                 loc="left", fontsize=16, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)

    # ------------------------------------------------------------------
    # Panel (b) — UNIFIED baseline, 4 bars
    # ------------------------------------------------------------------
    ax = axes[1]
    rows = panel_b_unified
    y = list(range(len(rows)))
    # Right-justify text labels at a fixed x so they never collide with
    # the data points or the y-axis tick labels. Place labels in the
    # clear region to the right of the most-positive marker.
    label_x = 0.78
    for yi, (lab, d, p, col, n_pts) in enumerate(rows):
        ax.plot([0, d], [yi, yi], color=col, lw=3.0, alpha=0.65)
        ax.scatter(d, yi, s=110, color=col, edgecolor="black", zorder=3)
        ax.text(label_x, yi,
                f"$d={d:+.3f}$  {sig_stars(p)}  ($n\\!=\\!{n_pts:,}$)",
                ha="right", va="center", color=col, fontsize=8.0,
                fontweight="bold")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 0.80)
    ax.set_xlabel("Cohen's $d$ vs pooled chr17+chr22 intron baseline\n"
                  "(negative = shallower than introns)")
    ax.set_title("(b) Unified baseline: regulatory effect sizes",
                 loc="left", fontsize=16, fontweight="bold",
                 fontfamily="Times New Roman", pad=12)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_2_shallowness.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_2_shallowness.pdf")
    plt.close(fig)


def main() -> None:
    setup_style()
    fig_2_shallowness_unified()
    print(f"wrote {OUT_DIR/'fig_2_shallowness.png'}")
    print(f"wrote {OUT_DIR/'fig_2_shallowness.pdf'}")


if __name__ == "__main__":
    main()
