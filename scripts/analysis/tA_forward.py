"""Tier A master forward — extracts per-position c(t) and H_t for all chr22 windows.

Outputs single parquet with columns:
  [window_idx, chrom, pos, label_code, c_t, H_t]

c(t) = settling depth with gamma_cos = 0.397, running-min envelope.
H_t = next-token Shannon entropy from h_norm @ W_E (fp32).

Per-layer cosine distance D_cos(ell, t) = 1 - cos(h_ell(t), h_norm(t)).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from tqdm import tqdm

# Resolve repo root and add src to path for constants
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.constants_evo2 import N_LAYERS, VOCAB_SIZE, BOS_OFFSET

GAMMA_COS = 0.397  # frozen calibration from chr22 paper
WINDOW = 6000
CENTRAL_START = 1500  # central 3kb anchor
CENTRAL_END = 4500

# Paths default to <repo>/data and <repo>/results; override via env vars if needed.
DATA_DIR = Path(os.environ.get("GDTR_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("GDTR_RESULTS_DIR", REPO_ROOT / "results"))
CHR22_FA = str(DATA_DIR / "external" / "chr22.fa")
CHR22_LABELS = str(DATA_DIR / "external" / "chr22_position_labels.npy")
CHR22_WINDOWS = str(DATA_DIR / "external" / "chr22_windows.tsv")
OUTPUT_PARQUET = str(RESULTS_DIR / "tier_A" / "tierA_chr22.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tA_forward")


def settling_depth_from_Dcos(D_cos: torch.Tensor, gamma: float = GAMMA_COS) -> torch.Tensor:
    """c(t) = smallest 1-based layer where running-min(D_cos) <= gamma.

    Args:
        D_cos: [n_layers, T]
        gamma: threshold.
    Returns:
        int64 [T], values in {1..n_layers}, =n_layers if never crossed.
    """
    L, T = D_cos.shape
    rmin = torch.cummin(D_cos, dim=0).values  # [L, T]
    below = rmin <= gamma  # [L, T]
    any_below = below.any(dim=0)
    first_idx = below.float().argmax(dim=0)  # 0-based first True
    c = torch.where(any_below, first_idx + 1, torch.tensor(L, dtype=first_idx.dtype))
    return c.to(torch.int64)


def extract_hidden_states_via_hooks(model, input_ids: torch.Tensor, layer_names: List[str]):
    """Forward through model with hooks to capture per-layer outputs.

    Returns dict {layer_name: [B, T, H] tensor}.
    """
    captured = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            # vortex blocks may return tuple (h, inference_params) or just h
            h = out[0] if isinstance(out, tuple) else out
            captured[name] = h.detach()
        return hook

    # Register hooks on sh.blocks[i] and sh.norm
    sh = model.model  # StripedHyena
    for name in layer_names:
        if name == "norm":
            mod = sh.norm
        elif name.startswith("blocks."):
            idx = int(name.split(".")[1])
            mod = sh.blocks[idx]
        else:
            raise ValueError(name)
        handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _ = model.forward(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


def compute_cos_distance_to_norm(hidden_states, n_layers: int, bos_offset: int) -> torch.Tensor:
    """D_cos(ell, t) = 1 - cos(h_ell, h_norm). Returns [n_layers, T_real] float32."""
    h_norm = hidden_states["norm"].float()[:, bos_offset:, :]  # [B, T, H]
    h_norm_n = F.normalize(h_norm, p=2, dim=-1)
    B, T, H = h_norm.shape
    D = torch.zeros((n_layers, T), dtype=torch.float32, device=h_norm.device)
    for ell in range(n_layers):
        h_l = hidden_states[f"blocks.{ell}"].float()[:, bos_offset:, :]
        h_l_n = F.normalize(h_l, p=2, dim=-1)
        cos = (h_l_n * h_norm_n).sum(dim=-1)
        d = (1.0 - cos).clamp(min=0.0).mean(dim=0)  # [T]
        D[ell] = d
    return D


def compute_entropy_from_norm(hidden_states, unembed_weight: torch.Tensor, bos_offset: int) -> torch.Tensor:
    """H_t = next-token Shannon entropy from h_norm @ unembed_weight^T.

    Args:
        hidden_states: dict with "norm" key.
        unembed_weight: [V, H] embedding weight tensor (tied with unembed).
        bos_offset: 0 for Evo 2.
    Returns:
        [T] float32 entropy in nats.
    """
    h_norm = hidden_states["norm"].float()[:, bos_offset:, :]  # [B, T, H]
    # logits = h_norm @ W_E^T; in Evo 2, unembed weight is the embedding (storage-tied)
    logits = h_norm @ unembed_weight.float().T  # [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    H = -(probs * log_probs).sum(dim=-1)  # [B, T]
    return H.mean(dim=0)  # [T]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="limit number of windows (for testing)")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument("--chunk-rows", type=int, default=500_000)
    args = p.parse_args()

    Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading Evo 2 7B...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")
    sh = model.model
    log.info("Evo 2 loaded; dtype=%s", next(sh.parameters()).dtype)

    # Locate embedding weight (tied with unembed). In StripedHyena, embedding_layer.weight.
    unembed_weight = sh.embedding_layer.weight.detach()
    log.info("unembed weight shape: %s dtype: %s", unembed_weight.shape, unembed_weight.dtype)

    # Load reference
    fa = Fasta(CHR22_FA, as_raw=True, sequence_always_upper=True)
    seq = str(fa["chr22"][:])
    chrom_len = len(seq)
    log.info("chr22 len=%d", chrom_len)

    # Load position labels
    labels = np.load(CHR22_LABELS)
    assert labels.shape[0] == chrom_len
    log.info("chr22 labels loaded; uniq=%s", np.unique(labels, return_counts=True))

    # Load windows table
    windows = pd.read_csv(CHR22_WINDOWS, sep="\t")
    log.info("chr22 windows: %d", len(windows))

    layer_names = [f"blocks.{i}" for i in range(N_LAYERS)] + ["norm"]

    # Subset windows for processing
    if args.end_idx is None:
        args.end_idx = len(windows)
    if args.limit is not None:
        args.end_idx = min(args.end_idx, args.start_idx + args.limit)
    windows_subset = windows.iloc[args.start_idx:args.end_idx].reset_index(drop=True)
    log.info("Processing windows [%d, %d): n=%d", args.start_idx, args.end_idx, len(windows_subset))

    # Pre-allocate output buffers (per-position rows in central 3kb of each window)
    # Each window contributes 3000 positions in the central region [start+1500, start+4500)
    # We dedupe via window_idx for now and merge later
    rows_window_idx = []
    rows_pos = []
    rows_label = []
    rows_c = []
    rows_H = []

    chunk_id = 0
    t0 = time.time()

    for i, row in tqdm(windows_subset.iterrows(), total=len(windows_subset), desc="forward"):
        wstart = int(row["start"])
        wend = int(row["end"])
        win_idx = int(row["window_idx"])

        # Slice sequence
        seq_window = seq[wstart:wend]
        if len(seq_window) != WINDOW:
            log.warning("window %d wrong length %d, skip", win_idx, len(seq_window))
            continue

        # Tokenize
        token_list = model.tokenizer.tokenize(seq_window)
        ids = torch.tensor(token_list, dtype=torch.int64).unsqueeze(0).cuda()

        try:
            hs = extract_hidden_states_via_hooks(model, ids, layer_names)
            D_cos = compute_cos_distance_to_norm(hs, N_LAYERS, BOS_OFFSET)  # [L, T] cuda fp32
            c_t = settling_depth_from_Dcos(D_cos, gamma=GAMMA_COS).cpu().numpy()  # [T]
            H_t = compute_entropy_from_norm(hs, unembed_weight, BOS_OFFSET).cpu().numpy()  # [T]
            # Free GPU mem
            for k in list(hs.keys()):
                del hs[k]
            del D_cos
            torch.cuda.empty_cache()
        except Exception as e:
            log.error("window %d forward failed: %s", win_idx, e)
            continue

        # Extract central 3kb only (positions [1500, 4500) within the window)
        positions_in_window = np.arange(CENTRAL_START, CENTRAL_END)
        genomic_positions = wstart + positions_in_window  # 0-based
        c_central = c_t[positions_in_window]
        H_central = H_t[positions_in_window]
        label_central = labels[genomic_positions]

        rows_window_idx.append(np.full(len(positions_in_window), win_idx, dtype=np.int32))
        rows_pos.append(genomic_positions.astype(np.int32))
        rows_label.append(label_central.astype(np.uint8))
        rows_c.append(c_central.astype(np.int16))
        rows_H.append(H_central.astype(np.float32))

        # Flush in chunks to avoid memory blowup
        total_rows = sum(len(x) for x in rows_window_idx)
        if total_rows >= args.chunk_rows:
            df = pd.DataFrame({
                "window_idx": np.concatenate(rows_window_idx),
                "pos": np.concatenate(rows_pos),
                "label_code": np.concatenate(rows_label),
                "c_t": np.concatenate(rows_c),
                "H_t": np.concatenate(rows_H),
            })
            chunk_path = OUTPUT_PARQUET.replace(".parquet", f"_chunk{chunk_id:04d}.parquet")
            df.to_parquet(chunk_path, index=False)
            log.info("flushed chunk %d: %d rows -> %s", chunk_id, len(df), chunk_path)
            chunk_id += 1
            rows_window_idx, rows_pos, rows_label, rows_c, rows_H = [], [], [], [], []

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(windows_subset) - i - 1) / rate / 60
            log.info("window %d/%d, %.2f win/s, ETA %.1f min", i + 1, len(windows_subset), rate, eta)

    # Flush remainder
    if rows_window_idx:
        df = pd.DataFrame({
            "window_idx": np.concatenate(rows_window_idx),
            "pos": np.concatenate(rows_pos),
            "label_code": np.concatenate(rows_label),
            "c_t": np.concatenate(rows_c),
            "H_t": np.concatenate(rows_H),
        })
        chunk_path = OUTPUT_PARQUET.replace(".parquet", f"_chunk{chunk_id:04d}.parquet")
        df.to_parquet(chunk_path, index=False)
        log.info("flushed final chunk %d: %d rows -> %s", chunk_id, len(df), chunk_path)

    log.info("DONE in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
