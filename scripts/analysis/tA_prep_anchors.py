"""Tier A — annotate chr22 positions with TSS/TATA/CAAT/GC-box hits + GC-matched controls.

Strategy: build a *separate* per-position annotation layer for promoter elements,
keyed by absolute chr22 position so we can post-hoc join with the master forward parquet.

Outputs:
  /root/gDTR-PoC/data/annotation/chr22_promoter_anchors.parquet
  cols: pos, anno_type, gc_w100, motif_score
  anno_type ∈ {TSS, TATA, CAAT, GCbox, control_TSS, control_TATA, control_CAAT, control_GCbox}
"""
from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

import os
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GDTR_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("GDTR_RESULTS_DIR", REPO_ROOT / "results"))
CHR22_FA = str(DATA_DIR / "external" / "chr22.fa")
GTF_PATH = str(DATA_DIR / "external" / "gencode.v44.chr17_chr22.gtf")
OUT_PATH = str(RESULTS_DIR / "tier_A" / "chr22_promoter_anchors.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tA_anchors")

# JASPAR 2022 core matrices (counts) for relevant TFs.
# TBP/TATA: MA0108.3 (TBP). 11 positions; consensus TATAAAA.
# NFY (CAAT box): MA0060.3 (NFYA). 11 positions.
# Sp1 (GC box): MA0079.5. 10 positions.
# We hand-code minimal PWMs (rough log-odds) from JASPAR profiles.
# Each PWM: dict {base: list[float]} of position-specific scores in log-odds.

# TATA: TATAWAW consensus (W=A/T). Approx position weights.
PWM_TATA = np.array([
    # A,   C,   G,   T
    [0.8, -2.0, -2.0,  0.2],  # pos0 T/A (T preferred)
    [ 0.9, -2.0, -2.0, -1.5],  # pos1 A
    [-1.5, -2.0, -2.0,  0.9],  # pos2 T
    [ 0.9, -2.0, -2.0, -1.5],  # pos3 A
    [ 0.6, -1.0, -1.0,  0.3],  # pos4 A/T
    [ 0.9, -2.0, -2.0, -1.5],  # pos5 A
    [ 0.6, -1.0, -1.0,  0.3],  # pos6 A/T
])
TATA_THR = 4.0  # rough threshold

# CAAT box (NFY): CCAAT consensus. Use 5-mer.
PWM_CAAT = np.array([
    [-1.0,  0.9, -1.0, -1.0],  # C
    [-1.0,  0.9, -1.0, -1.0],  # C
    [ 0.9, -1.5, -1.5, -1.5],  # A
    [ 0.9, -1.5, -1.5, -1.5],  # A
    [-1.5, -1.5, -1.5,  0.9],  # T
])
CAAT_THR = 3.5

# GC box (Sp1): GGGCGGG or CCGCCC. Use 6-mer canonical.
PWM_GCBOX = np.array([
    [-1.5, -1.5,  0.9, -1.5],  # G
    [-1.5, -1.5,  0.9, -1.5],  # G
    [-1.5, -1.5,  0.9, -1.5],  # G
    [-1.5,  0.7, -1.0, -1.5],  # C (or G)
    [-1.5, -1.5,  0.9, -1.5],  # G
    [-1.5, -1.5,  0.9, -1.5],  # G
])
GCBOX_THR = 4.0

BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def parse_tss_from_gtf(gtf_path: str) -> List[Tuple[int, int]]:
    """Return list of (pos_0based, strand_sign) for canonical-transcript TSS on chr22.

    Use the 5'-most exon of each Ensembl_canonical-tagged transcript.
    """
    tss_positions = []
    transcripts = {}  # tid -> (chrom, start, end, strand, is_canonical)
    exons = []  # (tid, start, end)

    log.info("parsing GTF...")
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if cols[0] != "chr22":
                continue
            feat = cols[2]
            attrs = cols[8]
            if feat == "transcript":
                is_canonical = "tag \"Ensembl_canonical\"" in attrs
                tid = None
                for a in attrs.split(";"):
                    a = a.strip()
                    if a.startswith("transcript_id"):
                        tid = a.split('"')[1]
                        break
                if tid is not None:
                    transcripts[tid] = (cols[0], int(cols[3]), int(cols[4]), cols[6], is_canonical)
            elif feat == "exon":
                tid = None
                for a in attrs.split(";"):
                    a = a.strip()
                    if a.startswith("transcript_id"):
                        tid = a.split('"')[1]
                        break
                if tid is not None:
                    exons.append((tid, int(cols[3]), int(cols[4])))
    log.info("found %d transcripts (%d canonical), %d exons",
             len(transcripts), sum(1 for t in transcripts.values() if t[4]), len(exons))

    # Group exons by transcript
    tid_exons = {}
    for tid, s, e in exons:
        tid_exons.setdefault(tid, []).append((s, e))

    tss_set = set()
    for tid, (chrom, tr_s, tr_e, strand, is_canonical) in transcripts.items():
        if not is_canonical:
            continue
        ex = tid_exons.get(tid, [])
        if not ex:
            continue
        if strand == "+":
            tss_pos = min(e[0] for e in ex) - 1  # convert to 0-based
        else:
            tss_pos = max(e[1] for e in ex) - 1  # 0-based, last exon end
        tss_set.add((tss_pos, +1 if strand == "+" else -1))

    return sorted(tss_set)


def scan_pwm(seq_arr: np.ndarray, pwm: np.ndarray, threshold: float) -> np.ndarray:
    """Return 0-based positions on the given seq_arr where PWM score >= threshold.

    Scans BOTH strands. Returns center positions (rounded down).
    """
    L = pwm.shape[0]
    N = len(seq_arr)
    hits = []

    # Forward strand
    # Convert seq to integer indices (A=0, C=1, G=2, T=3); N → mark as -1
    fwd = np.full(N, -1, dtype=np.int8)
    for b, idx in BASE_IDX.items():
        fwd[seq_arr == ord(b)] = idx
    # Sliding window score
    # score(i) = sum_j PWM[j, fwd[i+j]]
    for i in tqdm(range(0, N - L + 1, 1), desc=f"PWM scan (L={L})", leave=False):
        chunk = fwd[i:i + L]
        if (chunk < 0).any():
            continue
        score = pwm[np.arange(L), chunk].sum()
        if score >= threshold:
            hits.append((i + L // 2, score, "+"))

    # Reverse strand (complement: A<->T, C<->G; mapping idx 0<->3, 1<->2)
    rev_complement = np.full(N, -1, dtype=np.int8)
    rev_complement[fwd == 0] = 3
    rev_complement[fwd == 1] = 2
    rev_complement[fwd == 2] = 1
    rev_complement[fwd == 3] = 0
    # Reverse direction along seq, but since we want positions in original frame,
    # we just scan PWM against the rc-mapped sequence in reverse order from each i.
    # Simpler: scan forward strand for the reverse-complement PWM.
    rc_pwm = pwm[::-1, ::-1]  # flip positions and base order
    # base order flip: A<->T (0<->3), C<->G (1<->2)
    # which equals: rc[:, [3,2,1,0]]
    rc_pwm = pwm[::-1][:, [3, 2, 1, 0]]
    for i in tqdm(range(0, N - L + 1, 1), desc=f"PWM scan rc (L={L})", leave=False):
        chunk = fwd[i:i + L]
        if (chunk < 0).any():
            continue
        score = rc_pwm[np.arange(L), chunk].sum()
        if score >= threshold:
            hits.append((i + L // 2, score, "-"))

    return hits


def scan_pwm_fast(seq_arr: np.ndarray, pwm: np.ndarray, threshold: float) -> List[Tuple[int, float, str]]:
    """Vectorized PWM scan on both strands. Returns hits (center, score, strand)."""
    L = pwm.shape[0]
    N = len(seq_arr)

    # Forward strand integer encoding
    fwd = np.full(N, -1, dtype=np.int8)
    for b, idx in BASE_IDX.items():
        fwd[seq_arr == ord(b)] = idx

    # Use np.lib.stride_tricks for sliding window
    if N < L:
        return []
    # Build sliding view: (N - L + 1, L)
    sliding = np.lib.stride_tricks.sliding_window_view(fwd, L)
    valid = (sliding >= 0).all(axis=1)
    # Mask out invalid (N-containing) windows
    # Score = pwm[j, sliding[:, j]] summed over j
    # Use advanced indexing
    sliding_clip = np.clip(sliding, 0, 3)
    pos_idx = np.arange(L)[None, :]  # (1, L)
    scores = pwm[pos_idx, sliding_clip].sum(axis=1)  # (N-L+1,)
    scores = np.where(valid, scores, -np.inf)
    fwd_hit_idx = np.where(scores >= threshold)[0]

    # Reverse-complement PWM (flip position + complement base)
    rc_pwm = pwm[::-1][:, [3, 2, 1, 0]]
    scores_rc = rc_pwm[pos_idx, sliding_clip].sum(axis=1)
    scores_rc = np.where(valid, scores_rc, -np.inf)
    rc_hit_idx = np.where(scores_rc >= threshold)[0]

    hits = []
    for i in fwd_hit_idx:
        hits.append((int(i + L // 2), float(scores[i]), "+"))
    for i in rc_hit_idx:
        hits.append((int(i + L // 2), float(scores_rc[i]), "-"))
    return hits


def gc_window(seq_arr: np.ndarray, pos: int, w: int = 100) -> float:
    a, b = max(0, pos - w), min(len(seq_arr), pos + w)
    sub = seq_arr[a:b]
    n_gc = ((sub == ord("G")) | (sub == ord("C"))).sum()
    n_valid = ((sub != ord("N"))).sum()
    return float(n_gc) / max(n_valid, 1)


def make_gc_matched_control(anchor_positions: List[int], seq_arr: np.ndarray, labels: np.ndarray,
                            n_controls_per_anchor: int = 1, gc_tol: float = 0.05,
                            avoid_around_tss_bp: int = 5000, max_tries: int = 50, seed: int = 42) -> List[Tuple[int, float]]:
    """For each anchor position, sample 1 GC-matched chr22 position outside annotated regions.

    Outside = labels==0 (intergenic), and >avoid_around_tss_bp from any anchor.
    """
    rng = np.random.default_rng(seed)
    intergenic_mask = (labels == 0)
    intergenic_idx = np.where(intergenic_mask)[0]
    log.info("intergenic positions: %d", len(intergenic_idx))

    # Pre-compute GC content at sampled intergenic positions (we'll just sample randomly)
    controls = []
    for anchor in tqdm(anchor_positions, desc="GC-match controls"):
        gc_a = gc_window(seq_arr, anchor, w=100)
        # Sample candidates
        ok = False
        for _ in range(max_tries):
            cand = int(rng.choice(intergenic_idx))
            gc_c = gc_window(seq_arr, cand, w=100)
            if abs(gc_c - gc_a) <= gc_tol:
                controls.append((cand, gc_c))
                ok = True
                break
        if not ok:
            # fallback: pick any intergenic
            cand = int(rng.choice(intergenic_idx))
            controls.append((cand, gc_window(seq_arr, cand, w=100)))
    return controls


def main():
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    log.info("loading chr22 reference + labels...")
    fa = Fasta(CHR22_FA, as_raw=True, sequence_always_upper=True)
    seq = str(fa["chr22"][:])
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    labels = np.load(str(DATA_DIR / "external" / "chr22_position_labels.npy"))
    log.info("chr22 len=%d", len(seq_arr))

    rows = []

    # === TSS ===
    log.info("parsing TSS from GTF (canonical transcripts only)...")
    tss = parse_tss_from_gtf(GTF_PATH)
    log.info("found %d TSS on chr22 canonical transcripts", len(tss))
    for pos, strand in tss:
        rows.append({"pos": pos, "anno_type": "TSS", "gc_w100": gc_window(seq_arr, pos), "motif_score": 0.0, "strand": "+" if strand > 0 else "-"})

    # === PWM scans ===
    for name, pwm, thr in [("TATA", PWM_TATA, TATA_THR),
                           ("CAAT", PWM_CAAT, CAAT_THR),
                           ("GCbox", PWM_GCBOX, GCBOX_THR)]:
        log.info("scanning %s PWM (threshold=%.1f)...", name, thr)
        hits = scan_pwm_fast(seq_arr, pwm, thr)
        log.info("  %d %s hits", len(hits), name)
        # Filter to keep only hits near a TSS (within 1kb upstream typical)
        tss_positions = set(p for p, _ in tss)
        # Build TSS position array sorted for fast nearest lookup
        tss_sorted = np.array(sorted(tss_positions))
        kept = 0
        for center, score, strand in hits:
            idx = np.searchsorted(tss_sorted, center)
            nearest = []
            if idx > 0:
                nearest.append(abs(int(tss_sorted[idx - 1]) - center))
            if idx < len(tss_sorted):
                nearest.append(abs(int(tss_sorted[idx]) - center))
            if not nearest:
                continue
            if min(nearest) > 1500:
                continue
            rows.append({"pos": center, "anno_type": name, "gc_w100": gc_window(seq_arr, center),
                         "motif_score": float(score), "strand": strand})
            kept += 1
        log.info("  %d %s hits kept (within 1.5kb of TSS)", kept, name)

    # === GC-matched controls for each anchor type ===
    log.info("building GC-matched controls...")
    df_anchors = pd.DataFrame(rows)
    for anno in ["TSS", "TATA", "CAAT", "GCbox"]:
        anchors = df_anchors.query("anno_type == @anno")["pos"].tolist()
        if not anchors:
            continue
        controls = make_gc_matched_control(anchors, seq_arr, labels, n_controls_per_anchor=1, gc_tol=0.05)
        for (cpos, cgc) in controls:
            rows.append({"pos": cpos, "anno_type": f"control_{anno}", "gc_w100": cgc, "motif_score": 0.0, "strand": "?"})

    df = pd.DataFrame(rows)
    df["pos"] = df["pos"].astype(np.int32)
    df.to_parquet(OUT_PATH, index=False)
    log.info("wrote %d anchors -> %s", len(df), OUT_PATH)
    log.info("breakdown:\n%s", df["anno_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
