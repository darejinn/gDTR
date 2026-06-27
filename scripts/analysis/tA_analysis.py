"""Tier A analysis — A1 (5'UTR restriction-of-range), A2 (promoter elements),
A3 (effect-size unification), A4 (Fig 2 background sanity).

Inputs:
  /root/gDTR-PoC/results/tierA/tierA_chr22_chunk*.parquet  -- per-position c_t, H_t
  /root/gDTR-PoC/data/annotation/chr22_promoter_anchors.parquet -- TSS/PWM anchors

Outputs:
  results/tierA/A1_restriction_of_range.csv  -- per-context (n, c_mean, H_mean, H_std, H_iqr, rho)
  results/tierA/A2_promoter_elements.csv    -- per-promoter-class Cohen's d (vs intron, vs GC-matched control)
  results/tierA/A3_effect_sizes.csv         -- ε² and rank-biserial summary
  results/tierA/A4_fig2_baseline.csv        -- pooled chr22 intron baseline + Cohen's d per context
  results/tierA/figures/A1_sigma_vs_rho.{pdf,png}
  results/tierA/figures/A2_promoter_bar.{pdf,png}
  results/tierA/figures/Fig2_v2_unified.{pdf,png}

Usage:
  python tA_analysis.py --chunks-dir /root/gDTR-PoC/results/tierA \\
                        --anchors /root/gDTR-PoC/data/annotation/chr22_promoter_anchors.parquet \\
                        --out-dir results/tierA
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tA_analysis")

LABEL_CODES = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
               5: "splice_donor", 6: "splice_acceptor"}
LABEL_NAMES = list(LABEL_CODES.values())


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (independent samples)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd


def epsilon_squared_kw(H: float, n_total: int) -> float:
    """ε² effect size for Kruskal-Wallis."""
    return (H - 0) / (n_total - 1) if n_total > 1 else float("nan")


def rank_biserial_dunn(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial r for two-group Mann-Whitney/Dunn."""
    nx, ny = len(x), len(y)
    if nx < 1 or ny < 1:
        return float("nan")
    U, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1.0 - 2.0 * U / (nx * ny)


def load_chunks(chunks_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(f"{chunks_dir}/tierA_chr22_chunk*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no chunks in {chunks_dir}")
    log.info("loading %d chunks...", len(paths))
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    # dedupe: same pos may appear in adjacent windows (we kept central 3kb only, so should be unique)
    df = df.drop_duplicates(subset=["pos"], keep="first").reset_index(drop=True)
    df["label"] = df["label_code"].map(LABEL_CODES)
    log.info("loaded %d unique positions", len(df))
    return df


# === A1: Restriction-of-range diagnostic ===
def analysis_A1(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for code, name in LABEL_CODES.items():
        sub = df[df["label_code"] == code]
        if len(sub) < 100:
            continue
        c = sub["c_t"].to_numpy()
        H = sub["H_t"].to_numpy()
        rho, _ = stats.spearmanr(c, H)
        rows.append({
            "context": name,
            "n": len(sub),
            "c_mean": float(np.mean(c)),
            "c_median": float(np.median(c)),
            "c_sd": float(np.std(c, ddof=1)),
            "H_mean": float(np.mean(H)),
            "H_median": float(np.median(H)),
            "H_sd": float(np.std(H, ddof=1)),
            "H_iqr": float(np.percentile(H, 75) - np.percentile(H, 25)),
            "H_range": float(H.max() - H.min()),
            "rho_c_H": float(rho),
            "abs_rho": abs(float(rho)),
        })
    A1 = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    A1.to_csv(out_dir / "A1_restriction_of_range.csv", index=False)
    log.info("A1 table:\n%s", A1.to_string(index=False))

    # Scatter σ(H) vs |ρ|
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(A1["H_sd"], A1["abs_rho"], s=60)
    for _, r in A1.iterrows():
        ax.annotate(r["context"], (r["H_sd"], r["abs_rho"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"$\sigma(H_t)$  (per-context std of next-token entropy)")
    ax.set_ylabel(r"$|\rho(c, H)|$  (Spearman, abs)")
    ax.set_title("Restriction-of-range diagnostic")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "A1_sigma_vs_rho.pdf")
    plt.savefig(out_dir / "figures" / "A1_sigma_vs_rho.png", dpi=200)
    plt.close()
    return A1


# === A2: Promoter elements ===
def analysis_A2(df: pd.DataFrame, anchors: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # Get c_t at each anchor position
    pos_to_c = df.set_index("pos")["c_t"].to_dict()
    anchors = anchors.copy()
    anchors["c_t"] = anchors["pos"].map(pos_to_c)
    anchors = anchors.dropna(subset=["c_t"])
    log.info("anchors with c_t lookup: %d (lost %d to missing windows)", len(anchors), len(df) - len(pos_to_c))

    intron_c = df[df["label_code"] == 1]["c_t"].to_numpy()
    log.info("intron baseline: c_mean=%.3f n=%d", np.mean(intron_c), len(intron_c))

    rows = []
    for anno_type in ["TSS", "TATA", "CAAT", "GCbox"]:
        anchor_c = anchors[anchors["anno_type"] == anno_type]["c_t"].to_numpy()
        control_c = anchors[anchors["anno_type"] == f"control_{anno_type}"]["c_t"].to_numpy()
        if len(anchor_c) < 10:
            log.warning("too few anchors for %s: %d", anno_type, len(anchor_c))
            continue
        d_vs_intron = cohen_d(anchor_c, intron_c)
        d_vs_control = cohen_d(anchor_c, control_c) if len(control_c) >= 10 else float("nan")
        try:
            U, p = stats.mannwhitneyu(anchor_c, intron_c, alternative="two-sided")
        except Exception:
            U, p = float("nan"), float("nan")
        rows.append({
            "anno_type": anno_type,
            "n_anchor": len(anchor_c),
            "n_control": len(control_c),
            "anchor_c_mean": float(np.mean(anchor_c)),
            "control_c_mean": float(np.mean(control_c)) if len(control_c) else float("nan"),
            "intron_c_mean": float(np.mean(intron_c)),
            "d_vs_intron": d_vs_intron,
            "d_vs_gc_matched_control": d_vs_control,
            "mw_p_vs_intron": float(p),
        })
    A2 = pd.DataFrame(rows)
    A2.to_csv(out_dir / "A2_promoter_elements.csv", index=False)
    log.info("A2 table:\n%s", A2.to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(A2))
    w = 0.35
    ax.bar(x - w/2, A2["d_vs_intron"], w, label="vs. intron")
    ax.bar(x + w/2, A2["d_vs_gc_matched_control"], w, label="vs. GC-matched control")
    ax.set_xticks(x)
    ax.set_xticklabels(A2["anno_type"])
    ax.set_ylabel("Cohen's d")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Promoter element settling depth (chr22)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "A2_promoter_bar.pdf")
    plt.savefig(out_dir / "figures" / "A2_promoter_bar.png", dpi=200)
    plt.close()
    return A2


# === A3: Effect-size unification (recompute for the 7 canonical contexts) ===
def analysis_A3(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    intron_c = df[df["label_code"] == 1]["c_t"].to_numpy()
    rows = []
    contexts_pairwise = ["splice_donor", "splice_acceptor", "coding_exon", "5utr", "3utr", "intergenic"]
    for ctx in contexts_pairwise:
        code = [k for k, v in LABEL_CODES.items() if v == ctx][0]
        ctx_c = df[df["label_code"] == code]["c_t"].to_numpy()
        if len(ctx_c) < 100:
            continue
        d = cohen_d(ctx_c, intron_c)
        U, p_mw = stats.mannwhitneyu(ctx_c, intron_c, alternative="two-sided")
        r_rb = rank_biserial_dunn(ctx_c, intron_c)
        rows.append({
            "context": ctx,
            "n": len(ctx_c),
            "vs_intron_cohen_d": d,
            "mw_p": float(p_mw),
            "rank_biserial_r": r_rb,
        })

    # Kruskal-Wallis on all 7 contexts for §3.3-style omnibus reporting
    groups = []
    for code in [0, 1, 2, 3, 4, 5, 6]:
        g = df[df["label_code"] == code]["c_t"].to_numpy()
        if len(g) >= 100:
            groups.append(g)
    H, p_kw = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eps2 = epsilon_squared_kw(H, n_total)
    log.info("KW omnibus: H=%.3f p=%.2e ε²=%.4f (k=%d, n=%d)", H, p_kw, eps2, len(groups), n_total)

    A3 = pd.DataFrame(rows)
    A3.to_csv(out_dir / "A3_effect_sizes.csv", index=False)
    # Save KW stats
    with open(out_dir / "A3_kw_omnibus.json", "w") as f:
        json.dump({"H": float(H), "p": float(p_kw), "epsilon_squared": float(eps2),
                   "n_total": int(n_total), "k_groups": len(groups)}, f, indent=2)
    log.info("A3 table:\n%s", A3.to_string(index=False))
    return A3


# === A4: Fig 2 baseline sanity (chr22-only pooled intron baseline) ===
def analysis_A4(df: pd.DataFrame, anchors: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Re-derive paper's Fig 2 panel-(a) and panel-(b) under a SINGLE chr22 intron baseline.

    Panel (a): per-context bar with Cohen's d vs intron (chr22-only here since we don't pool chr17)
    Panel (b): regulatory annotations (cCRE-ELS, eQTL, GWAS) — we use the anchors file's TSS as
               proxy for promoter-class regulatory anchors. For cCRE-ELS the paper's bed file
               is needed from local (we can join later).
    """
    intron_c = df[df["label_code"] == 1]["c_t"].to_numpy()
    intron_mean = float(np.mean(intron_c))
    intron_sd = float(np.std(intron_c, ddof=1))

    # Per-context bar (panel a equivalent)
    rows = []
    for code, name in LABEL_CODES.items():
        ctx_c = df[df["label_code"] == code]["c_t"].to_numpy()
        if len(ctx_c) < 100:
            continue
        d = cohen_d(ctx_c, intron_c)
        rows.append({
            "context": name,
            "n": len(ctx_c),
            "c_mean": float(np.mean(ctx_c)),
            "c_sd": float(np.std(ctx_c, ddof=1)),
            "d_vs_intron": d,
        })

    # Add promoter elements with same baseline (panel b equivalent if cCRE-ELS file available)
    pos_to_c = df.set_index("pos")["c_t"].to_dict()
    anchors_local = anchors.copy()
    anchors_local["c_t"] = anchors_local["pos"].map(pos_to_c)
    anchors_local = anchors_local.dropna(subset=["c_t"])
    for anno_type in ["TSS", "TATA", "CAAT", "GCbox"]:
        sub = anchors_local[anchors_local["anno_type"] == anno_type]
        if len(sub) < 10:
            continue
        c = sub["c_t"].to_numpy()
        d = cohen_d(c, intron_c)
        rows.append({
            "context": f"promoter:{anno_type}",
            "n": len(c),
            "c_mean": float(np.mean(c)),
            "c_sd": float(np.std(c, ddof=1)),
            "d_vs_intron": d,
        })

    A4 = pd.DataFrame(rows)
    A4.to_csv(out_dir / "A4_fig2_baseline.csv", index=False)
    log.info("A4 (intron_mean=%.3f, intron_sd=%.3f):\n%s", intron_mean, intron_sd, A4.to_string(index=False))

    # Fig 2 v2: unified Cohen's d panel
    fig, ax = plt.subplots(figsize=(9, 4))
    A4_sorted = A4.sort_values("d_vs_intron", ascending=True)
    colors = ["#1f77b4" if c.startswith("promoter:") else
              "#d62728" if "splice" in c else
              "#7f7f7f" for c in A4_sorted["context"]]
    ax.barh(np.arange(len(A4_sorted)), A4_sorted["d_vs_intron"], color=colors)
    ax.set_yticks(np.arange(len(A4_sorted)))
    ax.set_yticklabels(A4_sorted["context"])
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Cohen's $d$ vs intron (chr22)")
    ax.set_title("Fig 2 v2 — unified chr22 intron baseline")
    for i, (_, r) in enumerate(A4_sorted.iterrows()):
        ax.text(r["d_vs_intron"], i, f" d={r['d_vs_intron']:.3f} (n={r['n']:,})",
                va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "Fig2_v2_unified.pdf")
    plt.savefig(out_dir / "figures" / "Fig2_v2_unified.png", dpi=200)
    plt.close()
    return A4


def main():
    import os
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = Path(os.environ.get("GDTR_RESULTS_DIR", repo_root / "results")) / "tier_A"
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", default=str(results_dir))
    p.add_argument("--anchors", default=str(results_dir / "chr22_promoter_anchors.parquet"))
    p.add_argument("--out-dir", default=str(results_dir / "analysis"))
    args = p.parse_args()

    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_chunks(args.chunks_dir)
    anchors = pd.read_parquet(args.anchors)
    log.info("anchors: %d", len(anchors))

    log.info("=== A1 Restriction-of-range ===")
    analysis_A1(df, out)

    log.info("=== A2 Promoter elements ===")
    analysis_A2(df, anchors, out)

    log.info("=== A3 Effect-size unification ===")
    analysis_A3(df, out)

    log.info("=== A4 Fig 2 background sanity ===")
    analysis_A4(df, anchors, out)

    log.info("DONE")


if __name__ == "__main__":
    main()
