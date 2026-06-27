# Reproduction Guide

## Two paths

This repository supports two reproduction paths.

1. **Inspect committed tables.** All paper-cited numbers come from CSV/JSON files under `results/tier_A/`. No GPU is required and no external data download is needed. Start here if you want to verify a specific number in the paper.
2. **Rerun the pipeline.** Recompute c(t) and the downstream tables on chr22 and chr17 from scratch. Requires an NVIDIA H100 (80 GB) or H200 (141 GB), the Evo 2 7B weights (about 14 GB), and the external annotations listed in §2. The master forward pass takes roughly 60 minutes on an H100.

The figure regeneration scripts under `scripts/figures/` depend on a small number of upstream metadata artifacts (`results/figures_v3/fig_v9_meta.json`, `results/phase4/per_model_summary.json`) that come from Phase-1 through Phase-4 of the historical pipeline and are not committed. They are included so the plotting code is readable; the numbers behind every figure are in `results/tier_A/` and can be inspected without running the regeneration.

## 1. Environment

Tested on Ubuntu 22.04 with CUDA 12.8.

```
bash scripts/env_setup.sh
```

Installs:

- torch 2.7.1+cu128
- transformer_engine 2.16.0 (built from source against the installed torch)
- flash-attn 2.8.3.post1 (rebuilt against torch 2.7.1)
- evo2 0.3.0 (loads `arcinstitute/evo2_7b` at HF revision `bda0089f92582d5baabf0f22d9fc85f3588f6b58`)
- biopython, pyfaidx, pyBigWig, gffutils, statannotations, pyarrow

The historical paper run used torch 2.4.1+cu124 with transformer_engine 2.14.0 on an H200. Numerical results agree across both stacks within rounding.

## 2. Data

Download external annotations into `data/external/`. URLs and sizes are in `data/external/README.md`.

Reference genome: GRCh38 primary assembly (UCSC). Only `chr17.fa` and `chr22.fa` are used end-to-end.

Annotations:

- GENCODE v44 GTF (basic, primary annotation): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/
- ENCODE SCREEN v3/v4 cCRE catalog (ELS subset on chr22)
- GTEx v8 single-tissue cis-eQTL (Whole Blood, chr22 significant variant-gene pairs)
- GWAS Catalog v1.0 (chr22 SNP positions)
- ClinVar 2026-04-18 VCF (15 cancer-associated genes)

## 3. Pipeline

```
# Window selection and annotations
python scripts/prep/prep_chr22_windows.py
python scripts/prep/prep_chr17_windows.py
python scripts/analysis/tA_prep_anchors.py

# Master forward pass (12,978 chr22 windows; about 60 minutes on H100)
python scripts/analysis/tA_forward.py

# Analyses
python scripts/analysis/tA_analysis.py           # A1/A2/A3/A4 (self-consistent c)
python scripts/analysis/tA_analysis_optionA.py   # paper canonical c joined with fresh H_t
python scripts/analysis/compute_rank_biserial.py # nonsense vs missense Dunn effect
python scripts/analysis/compute_fig2_unified_d.py # Fig 2 panel (b) unified baseline d
```

Output tables land in `results/tier_A/`. They should match the committed reference tables within rounding.

## 4. Figures

```
python scripts/figures/regen_fig_shallowness_unified.py
python scripts/figures/regen_fig_A4_splice_fine_local.py
# and other regen_fig_*.py scripts
```

These scripts expect `results/figures_v3/fig_v9_meta.json` and `results/phase4/per_model_summary.json`, which come from Phase-1 through Phase-4 of the historical pipeline and are not committed. The numbers behind every figure cited in the paper are in `results/tier_A/`.

## 5. Paper build

```
cd paper
pdflatex -interaction=nonstopmode gdtr_paper_ICML_3.tex
bibtex gdtr_paper_ICML_3
pdflatex -interaction=nonstopmode gdtr_paper_ICML_3.tex
pdflatex -interaction=nonstopmode gdtr_paper_ICML_3.tex
```

## 6. Expected timings

| Stage | Time |
|---|---|
| Env setup (first time) | ~30 min |
| chr22 + chr17 window prep | ~15 s |
| TSS / PWM anchor prep | ~15 s |
| Master forward (chr22, 12,978 windows) | ~60 min on H100 |
| Tier-A analyses | ~1 min total |
| Figure regeneration | ~5 s each |
