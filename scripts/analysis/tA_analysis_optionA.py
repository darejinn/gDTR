"""Tier A analysis — Option A: use PAPER's cached c(t) + my fresh H_t.

Why this script exists
----------------------
My fresh tA_forward.py used cos(h_ell, h_norm) as the lens reference, following
the paper TEXT. But the paper CODE (TDiG/scripts/15_chr22_forward.py:130-145)
uses cos(h_ell, h_29) — the L*=29 tap itself — as the primary reference
(saved as cos_refA). The cached chr22_position_c.npy in results/phase2.4/ is
derived from cos_refA, NOT cos to h_norm.

Consequence of using h_norm in fresh forward: ~2.4-layer systematic offset
(intron c̄ = 30.14 vs paper 27.72), heavy saturation at c=32, and ρ(c, H)
collapse for 5'UTR (-0.036 vs paper +0.41).

Option A fixes this by using PAPER's cached c(t) values (the canonical
refA-derived array) for the layer-level metric while keeping my fresh
H_t values (computed from h_norm @ W_E in the same Tier A forward) for
entropy. Both quantities index by absolute chr22 position so they merge cleanly.

Inputs (server):
  /root/gDTR-PoC/data/annotation/chr22_position_c_PAPER.npy    (paper refA c)
  /root/gDTR-PoC/data/annotation/chr22_position_labels.npy
  /root/gDTR-PoC/data/annotation/chr22_promoter_anchors.parquet
  /root/gDTR-PoC/results/tierA/tierA_chr22_chunk*.parquet      (my pos, label_code, H_t)

Outputs:
  /root/gDTR-PoC/results/tierA/analysis_optionA/A1_restriction_of_range.csv
  /root/gDTR-PoC/results/tierA/analysis_optionA/A2_promoter_elements.csv
  /root/gDTR-PoC/results/tierA/analysis_optionA/A4_fig2_baseline.csv
"""
from __future__ import annotations
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tA_optionA")

import os
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GDTR_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("GDTR_RESULTS_DIR", REPO_ROOT / "results")) / "tier_A"
PAPER_C = DATA_DIR / "external" / "chr22_position_c_PAPER.npy"
LABELS = DATA_DIR / "external" / "chr22_position_labels.npy"
ANCHORS = RESULTS_DIR / "chr22_promoter_anchors.parquet"
CHUNKS_DIR = RESULTS_DIR
OUT_DIR = RESULTS_DIR / "analysis_optionA"

LABEL_CODES = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr", 4: "3utr",
               5: "splice_donor", 6: "splice_acceptor"}


def cohen_d_indep(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def load_my_Ht(chunks_dir: Path) -> pd.DataFrame:
    """Return DataFrame with [pos, H_t, label_code] from my Tier A chunks (central 3kb per window)."""
    paths = sorted(glob.glob(str(chunks_dir / "tierA_chr22_chunk*.parquet")))
    log.info("loading %d chunks for H_t...", len(paths))
    dfs = [pd.read_parquet(p, columns=["pos", "H_t", "label_code"]) for p in paths]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["pos"], keep="first")
    log.info("unique positions with H_t: %d", len(df))
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("loading paper's c(t) (refA-derived)...")
    paper_c = np.load(PAPER_C)
    log.info("  shape=%s dtype=%s nan=%d non-nan mean=%.4f",
             paper_c.shape, paper_c.dtype, np.isnan(paper_c).sum(), np.nanmean(paper_c))

    labels = np.load(LABELS)
    log.info("labels loaded: %s", dict(zip(*np.unique(labels, return_counts=True))))

    ht_df = load_my_Ht(CHUNKS_DIR)

    # Lookup paper_c at my positions
    pos = ht_df["pos"].to_numpy()
    paper_c_at_pos = paper_c[pos]
    label_at_pos = labels[pos]
    ht_df["c_t_paper"] = paper_c_at_pos
    ht_df["label_code_from_full"] = label_at_pos
    # Drop NaN c values
    n_before = len(ht_df)
    ht_df = ht_df[~np.isnan(ht_df["c_t_paper"])].reset_index(drop=True)
    log.info("after dropping NaN paper_c: %d (lost %d)", len(ht_df), n_before - len(ht_df))

    # === A1 — restriction-of-range with paper c ===
    log.info("=== A1: restriction-of-range (paper c + my H_t) ===")
    rows = []
    for code, name in LABEL_CODES.items():
        sub = ht_df[ht_df["label_code_from_full"] == code]
        if len(sub) < 100:
            continue
        c = sub["c_t_paper"].to_numpy().astype(float)
        H = sub["H_t"].to_numpy().astype(float)
        rho, _ = stats.spearmanr(c, H)
        rows.append({
            "context": name,
            "n": len(sub),
            "c_mean": float(np.mean(c)),
            "c_median": float(np.median(c)),
            "c_sd": float(np.std(c, ddof=1)),
            "H_mean": float(np.mean(H)),
            "H_sd": float(np.std(H, ddof=1)),
            "H_iqr": float(np.percentile(H, 75) - np.percentile(H, 25)),
            "H_range": float(H.max() - H.min()),
            "rho_c_H": float(rho),
            "abs_rho": abs(float(rho)),
        })
    A1 = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    A1.to_csv(OUT_DIR / "A1_restriction_of_range.csv", index=False)
    log.info("A1:\n%s", A1.to_string(index=False))

    # === A2 — promoter elements (paper c at TSS/TATA/CAAT/GC anchors) ===
    log.info("=== A2: promoter elements (paper c at anchors) ===")
    anchors = pd.read_parquet(ANCHORS)
    anchors["c_paper"] = anchors["pos"].apply(lambda p: paper_c[int(p)])
    anchors = anchors[~np.isnan(anchors["c_paper"])].reset_index(drop=True)
    log.info("anchors after NaN drop: %d", len(anchors))

    intron_c = paper_c[(labels == 1) & ~np.isnan(paper_c)]
    log.info("intron baseline (paper c, chr22 only): n=%d mean=%.4f sd=%.4f",
             len(intron_c), intron_c.mean(), intron_c.std(ddof=1))

    rows = []
    for anno in ["TSS", "TATA", "CAAT", "GCbox"]:
        anchor_c = anchors[anchors["anno_type"] == anno]["c_paper"].to_numpy().astype(float)
        ctrl_c = anchors[anchors["anno_type"] == f"control_{anno}"]["c_paper"].to_numpy().astype(float)
        if len(anchor_c) < 10:
            continue
        d_vs_intron = cohen_d_indep(anchor_c, intron_c)
        d_vs_ctrl = cohen_d_indep(anchor_c, ctrl_c) if len(ctrl_c) >= 10 else float("nan")
        U, p = stats.mannwhitneyu(anchor_c, intron_c, alternative="two-sided")
        rows.append({
            "anno_type": anno,
            "n_anchor": int(len(anchor_c)),
            "n_control": int(len(ctrl_c)),
            "anchor_c_mean": float(anchor_c.mean()),
            "control_c_mean": float(ctrl_c.mean()) if len(ctrl_c) else float("nan"),
            "intron_c_mean": float(intron_c.mean()),
            "d_vs_intron": float(d_vs_intron),
            "d_vs_gc_matched_control": float(d_vs_ctrl),
            "mw_p_vs_intron": float(p),
        })
    A2 = pd.DataFrame(rows)
    A2.to_csv(OUT_DIR / "A2_promoter_elements.csv", index=False)
    log.info("A2:\n%s", A2.to_string(index=False))

    # === A4 — per-context baseline (paper c, chr22-only) for sanity ===
    log.info("=== A4: per-context baseline (paper c chr22) ===")
    rows = []
    for code, name in LABEL_CODES.items():
        mask = (labels == code) & ~np.isnan(paper_c)
        c = paper_c[mask]
        if len(c) < 100:
            continue
        d = cohen_d_indep(c, intron_c)
        rows.append({
            "context": name,
            "n": int(len(c)),
            "c_mean": float(c.mean()),
            "c_sd": float(c.std(ddof=1)),
            "d_vs_intron_chr22": float(d),
        })
    A4 = pd.DataFrame(rows)
    A4.to_csv(OUT_DIR / "A4_fig2_baseline.csv", index=False)
    log.info("A4:\n%s", A4.to_string(index=False))

    log.info("DONE — outputs at %s", OUT_DIR)


if __name__ == "__main__":
    main()
