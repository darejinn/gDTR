"""Compute Fig 2 Cohen's d values under a SINGLE pooled chr17+chr22 intron baseline.

Inputs (local):
  /Users/yoonjincho/Project/ICML/results/phase2.4/chr22_position_c.npy
  /Users/yoonjincho/Project/ICML/results/phase2.4/chr17_position_c.npy
  ./data_local/chr22_position_labels.npy   (codebook: 1=intron, 5=splice_donor, 6=splice_acceptor)
  ./data_local/chr17_position_labels.npy
  ./data_local/ccre_els_chr22.bed          (ENCODE cCRE v4 ELS subset, chr22)
  ./data_local/eqtl_chr22.bed              (optional)
  ./data_local/gwas_chr22.bed              (optional)

Output: prints d_vs_pooled_intron and (n_anchor, anchor_mean, intron_mean, pooled_SD)
for each annotation, ready to paste into 0627_3.tex.

Cohen's d (independent samples, pooled SD):
  d = (mean_anchor - mean_intron) / sqrt(((n_a-1)*var_a + (n_i-1)*var_i) / (n_a + n_i - 2))
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

# Repo-relative defaults; override with GDTR_DATA_DIR / GDTR_RESULTS_DIR.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GDTR_DATA_DIR", REPO_ROOT / "data")) / "external"
RESULTS_DIR = Path(os.environ.get("GDTR_RESULTS_DIR", REPO_ROOT / "results")) / "tier_A"
ROOT = REPO_ROOT
C22 = os.environ.get("GDTR_CHR22_C", str(DATA / "chr22_position_c.npy"))
C17 = os.environ.get("GDTR_CHR17_C", str(DATA / "chr17_position_c.npy"))
LAB22 = DATA / "chr22_position_labels.npy"
LAB17 = DATA / "chr17_position_labels.npy"


def cohen_d_indep(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def load_bed_mask(bed_path: Path, chrom_len: int, chrom: str = "chr22") -> np.ndarray:
    mask = np.zeros(chrom_len, dtype=bool)
    if not bed_path.exists():
        print(f"  WARN: {bed_path} missing")
        return mask
    n = 0
    with bed_path.open() as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#") or ln.startswith("track"):
                continue
            parts = ln.split("\t")
            if len(parts) < 3 or parts[0] != chrom:
                continue
            try:
                a, b = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            a = max(0, a); b = min(chrom_len, b)
            if 0 <= a < b <= chrom_len:
                mask[a:b] = True
                n += 1
    print(f"  {bed_path.name}: {n} records, {mask.sum():,} bp")
    return mask


def main() -> None:
    print("loading per-position c(t)...")
    c22 = np.load(C22)
    c17 = np.load(C17)
    print(f"  chr22 c: shape={c22.shape}, NaN count={np.isnan(c22).sum():,}")
    print(f"  chr17 c: shape={c17.shape}, NaN count={np.isnan(c17).sum():,}")

    lab22 = np.load(LAB22)
    lab17 = np.load(LAB17)
    print(f"  chr22 labels: {dict(zip(*np.unique(lab22, return_counts=True)))}")
    print(f"  chr17 labels: {dict(zip(*np.unique(lab17, return_counts=True)))}")

    valid22 = ~np.isnan(c22)
    valid17 = ~np.isnan(c17)

    # === Pooled chr17+chr22 intron baseline ===
    intron22 = c22[(lab22 == 1) & valid22]
    intron17 = c17[(lab17 == 1) & valid17]
    intron_pool = np.concatenate([intron22, intron17])
    intron_mean = float(intron_pool.mean())
    intron_sd = float(intron_pool.std(ddof=1))
    print(f"\n=== POOLED chr17+chr22 INTRON BASELINE ===")
    print(f"  n_chr22 = {len(intron22):,}, n_chr17 = {len(intron17):,}, n_pool = {len(intron_pool):,}")
    print(f"  mean c = {intron_mean:.4f}   (expected ≈ 27.7156)")
    print(f"  sd c   = {intron_sd:.4f}")

    results = {}

    # === splice_donor pooled chr17+chr22 ===
    donor22 = c22[(lab22 == 5) & valid22]
    donor17 = c17[(lab17 == 5) & valid17]
    donor_pool = np.concatenate([donor22, donor17])
    d_donor = cohen_d_indep(donor_pool, intron_pool)
    print(f"\n=== splice donor (pooled chr17+chr22) ===")
    print(f"  n = {len(donor_pool):,}, mean = {donor_pool.mean():.4f}")
    print(f"  Cohen's d vs pooled intron = {d_donor:+.4f}   (expected ≈ -0.3575 from fig_v9_meta)")
    results["splice_donor"] = {"n": int(len(donor_pool)), "mean_c": float(donor_pool.mean()), "d": d_donor}

    # === splice_acceptor pooled chr17+chr22 ===
    acc22 = c22[(lab22 == 6) & valid22]
    acc17 = c17[(lab17 == 6) & valid17]
    acc_pool = np.concatenate([acc22, acc17])
    d_acc = cohen_d_indep(acc_pool, intron_pool)
    print(f"\n=== splice acceptor (pooled chr17+chr22) ===")
    print(f"  n = {len(acc_pool):,}, mean = {acc_pool.mean():.4f}")
    print(f"  Cohen's d vs pooled intron = {d_acc:+.4f}   (expected ≈ -0.3397 from fig_v9_meta)")
    results["splice_acceptor"] = {"n": int(len(acc_pool)), "mean_c": float(acc_pool.mean()), "d": d_acc}

    # === cCRE-ELS chr22 only ===
    print("\nloading cCRE-ELS chr22 mask...")
    ccre_mask = load_bed_mask(DATA / "ccre_els_chr22.bed", chrom_len=len(c22))
    if ccre_mask.any():
        ccre_c = c22[ccre_mask & valid22]
        d_ccre = cohen_d_indep(ccre_c, intron_pool)
        print(f"  n = {len(ccre_c):,}, mean = {ccre_c.mean():.4f}")
        print(f"  Cohen's d vs pooled intron = {d_ccre:+.4f}   (chr22-only-bg was -0.1903; v4 catalog)")
        results["cCRE_ELS"] = {"n": int(len(ccre_c)), "mean_c": float(ccre_c.mean()), "d": d_ccre}

    # === eQTL chr22 (optional) ===
    print("\nloading GTEx eQTL chr22 mask...")
    eqtl_mask = load_bed_mask(DATA / "eqtl_chr22.bed", chrom_len=len(c22))
    if eqtl_mask.any():
        eqtl_c = c22[eqtl_mask & valid22]
        d_eqtl = cohen_d_indep(eqtl_c, intron_pool)
        print(f"  n = {len(eqtl_c):,}, mean = {eqtl_c.mean():.4f}")
        print(f"  Cohen's d vs pooled intron = {d_eqtl:+.4f}   (chr22-only-bg was -0.0224)")
        results["eQTL"] = {"n": int(len(eqtl_c)), "mean_c": float(eqtl_c.mean()), "d": d_eqtl}

    # === GWAS chr22 (optional) ===
    print("\nloading GWAS Catalog chr22 mask...")
    gwas_mask = load_bed_mask(DATA / "gwas_chr22.bed", chrom_len=len(c22))
    if gwas_mask.any():
        gwas_c = c22[gwas_mask & valid22]
        d_gwas = cohen_d_indep(gwas_c, intron_pool)
        print(f"  n = {len(gwas_c):,}, mean = {gwas_c.mean():.4f}")
        print(f"  Cohen's d vs pooled intron = {d_gwas:+.4f}   (chr22-only-bg was -0.0395)")
        results["GWAS"] = {"n": int(len(gwas_c)), "mean_c": float(gwas_c.mean()), "d": d_gwas}

    # Write summary
    summary = {
        "intron_baseline_pooled": {"mean": intron_mean, "sd": intron_sd, "n": int(len(intron_pool))},
        "results": results,
    }
    out = RESULTS_DIR / "fig2_unified_d_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary: {out}")


if __name__ == "__main__":
    main()
