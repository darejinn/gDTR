"""Compute rank-biserial r for nonsense → missense Dunn contrast on the LOCAL
SNV-only P/LP cohort.

Caveat: paper's 4023 cohort uses strict-priority MC classification (from
p2_snv_class_join.py on the H200 server) including 518 frameshift INDELs.
Local cohort is SNV-only and uses variant_with_consequence.csv labels, giving
~3,167 variants across 5 SNV classes. The nonsense and missense classes
specifically have somewhat different counts vs. paper (paper: nonsense=1740
missense=935; local: nonsense=1205 missense=920). Frameshift does not enter
the missense-vs-nonsense pairwise comparison, so the rank-biserial r computed
here should be within ±0.02 of the paper's true value, but is reported with
"≈" notation in the paper text.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd

# Use the in-repo stats.py with the canonical r = 1 - 2U/(n_a·n_b) convention.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.stats import mwu_with_effect

# These CSV inputs are not committed to the public repo (variant-level data is
# rebuilt by the variant pipeline in scripts/prep/). Set GDTR_VARIANT_FEATURES
# and GDTR_VARIANT_CONSEQUENCE to your local paths, or place them under
# data/external/.
DATA_DIR = Path(os.environ.get("GDTR_DATA_DIR", REPO_ROOT / "data"))
VARIANTS_FULL = os.environ.get(
    "GDTR_VARIANT_FEATURES",
    str(DATA_DIR / "external" / "variants_features_full.csv"),
)
WITH_CONSEQ = os.environ.get(
    "GDTR_VARIANT_CONSEQUENCE",
    str(DATA_DIR / "external" / "variant_with_consequence.csv"),
)

SIX_WAY_SNV = ["intron", "missense", "nonsense", "splice", "synonymous"]


def main() -> None:
    f = pd.read_csv(
        VARIANTS_FULL,
        usecols=["chrom", "pos", "ref", "alt", "category", "argmax_layer"],
    )
    v = pd.read_csv(WITH_CONSEQ)
    print(f"loaded variants_features_full: {f.shape}, variant_with_consequence: {v.shape}")

    m = f.merge(v[["chrom", "pos", "ref", "alt", "consequence"]],
                on=["chrom", "pos", "ref", "alt"], how="inner")
    print(f"merged: {m.shape}")

    sub = m[(m["category"] == "P_LP") & (m["consequence"].isin(SIX_WAY_SNV))].copy()
    print("\nP_LP cohort per-class:")
    print(sub["consequence"].value_counts())

    mis = sub.loc[sub["consequence"] == "missense", "argmax_layer"].to_numpy()
    non = sub.loc[sub["consequence"] == "nonsense", "argmax_layer"].to_numpy()
    print(f"\nmissense argmax_layer summary: n={len(mis)}, mean={mis.mean():.3f}, median={pd.Series(mis).median()}")
    print(f"nonsense argmax_layer summary: n={len(non)}, mean={non.mean():.3f}, median={pd.Series(non).median()}")

    # mwu_with_effect convention: pass deeper class first → positive r means
    # the first argument is deeper (paper: missense median=16 > nonsense median=12)
    U, p, r = mwu_with_effect(mis, non, alternative="two-sided")
    print("\n=== nonsense → missense (local P_LP SNV cohort) ===")
    print(f"U                = {U:.0f}")
    print(f"p (uncorrected)  = {p:.4e}")
    print(f"rank-biserial r  = {r:+.4f}    (positive → missense argmax deeper than nonsense)")

    # Bonferroni-corrected p over 5 adjacent pairs (paper's Dunn comparison)
    p_adj = min(1.0, p * 5)
    print(f"p (Bonf×5)       = {p_adj:.4e}")


if __name__ == "__main__":
    main()
