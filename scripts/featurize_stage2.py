"""
Stage 2 — Step 1: Featurize AlphaFold PDB files → backbone geometry dicts.

Reads usable proteins from proteins.csv, featurizes each PDB file in parallel,
and saves results to chunked pkl files. Run encode_stage2.py afterwards.

Output: /data/steven/ProteinChamaleon/encoded/featurized/chunk_XXXX.pkl
  Each file contains a dict with keys:
    structures   — list of geometry dicts (or None on failure)
    accessions   — list of UniProt accession IDs
    function_text— list of function descriptions
    organism     — list of organism names

Resume-safe: skips chunk files that already exist.

Usage:
    conda activate GeoBPE-6
    python scripts/featurize_stage2.py
    python scripts/featurize_stage2.py --workers 32 --chunk-size 5000
"""

import argparse
import functools
import logging
import multiprocessing
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/home/steven/PT-BPE")

from foldingdiff.angles_and_coords import (
    canonical_distances_and_dihedrals,
    extract_backbone_coords,
    extract_backbone_residue_idxes,
    extract_c_beta_coords,
    extract_side_chain_coords,
    extract_aa_seq,
    EXHAUSTIVE_ANGLES,
    EXHAUSTIVE_DISTS,
)
from foldingdiff.datasets import featurize_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("featurize_stage2")

PROTEINS_CSV   = Path("/home/steven/ProteinChamaleon-Dataset/output/proteins.csv")
STRUCTURES_DIR = Path("/data/steven/ProteinChamaleon/structures")
OUT_DIR        = Path("/data/steven/ProteinChamaleon/encoded/featurized")

MIN_FUNC_LEN = 10
_FUNC_DENYLIST = {"not yet known", "unknown", "unknown function", "function unknown"}

_FEATURIZER_BASE = functools.partial(
    featurize_one,
    pfunc=functools.partial(
        canonical_distances_and_dihedrals,
        distances=EXHAUSTIVE_DISTS,
        angles=EXHAUSTIVE_ANGLES,
    ),
    coords_pfunc=functools.partial(extract_backbone_coords, atoms=["CA"]),
    full_coord_pfunc=functools.partial(extract_backbone_coords, atoms=["N", "CA", "C"]),
    full_atom_idx_map=functools.partial(extract_backbone_residue_idxes, atoms=["N", "CA", "C"]),
    side_chain_coords_pfunc=extract_side_chain_coords,
    aa_seq_func=extract_aa_seq,
    c_beta_func=extract_c_beta_coords,
)


def _featurize_picklable(fname: str):
    """Featurize one PDB; convert DataFrame → numpy dict for safe IPC pickling."""
    result = _FEATURIZER_BASE(fname)
    if result is None:
        return None
    result["angles"] = {col: result["angles"][col].to_numpy()
                        for col in result["angles"].columns}
    return result


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", PROTEINS_CSV)
    df = pd.read_csv(PROTEINS_CSV)
    has_func = (
        df["function_text"].notna()
        & (df["function_text"].str.len() >= MIN_FUNC_LEN)
        & ~df["function_text"].str.lower().str.strip().isin(_FUNC_DENYLIST)
    )
    has_struct = df["accession"].apply(
        lambda acc: (STRUCTURES_DIR / f"{acc}.pdb").exists()
    )
    df = df[has_func & has_struct].reset_index(drop=True)
    logger.info("Usable proteins: %d", len(df))

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limiting to %d", len(df))

    fnames       = [str(STRUCTURES_DIR / f"{acc}.pdb") for acc in df["accession"]]
    chunk_size   = args.chunk_size
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    pending = [i for i in range(total_chunks)
               if not (OUT_DIR / f"chunk_{i:04d}.pkl").exists()]

    if not pending:
        logger.info("All %d chunks already done.", total_chunks)
        return

    logger.info("%d / %d chunks to featurize", len(pending), total_chunks)

    # spawn avoids fork-CoW OOM: workers start fresh with no inherited state.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for chunk_idx in pending:
            start     = chunk_idx * chunk_size
            chunk_df  = df.iloc[start : start + chunk_size]
            chunk_fnames = fnames[start : start + chunk_size]
            ckpt      = OUT_DIR / f"chunk_{chunk_idx:04d}.pkl"

            logger.info("Chunk %d/%d  (%d proteins)...",
                        chunk_idx + 1, total_chunks, len(chunk_fnames))

            structures = list(tqdm(
                pool.imap(_featurize_picklable, chunk_fnames, chunksize=20),
                total=len(chunk_fnames),
                desc=f"featurize {chunk_idx + 1}/{total_chunks}",
            ))

            with open(ckpt, "wb") as f:
                pickle.dump({
                    "structures":    structures,
                    "accessions":    chunk_df["accession"].tolist(),
                    "function_text": chunk_df["function_text"].tolist(),
                    "organism":      chunk_df["organism"].tolist(),
                }, f)
            logger.info("Saved %s", ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",    type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()
    main(args)
