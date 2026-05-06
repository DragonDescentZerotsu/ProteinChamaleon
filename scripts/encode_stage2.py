"""
Stage 2 — Step 2: Encode featurized structures → PT-BPE token arrays.

Loads BPE checkpoint ONCE in the main process, then forks a pool of workers
that share it via copy-on-write. This avoids the N×6GB RAM cost of the old
per-process approach — total RAM is ~6GB (shared BPE) + N×~1GB (chunk data).

Input:  featurized/chunk_XXXX.pkl  (from featurize_stage2.py)
Output: stage2.npz
  - token_ids:     object array of int32 arrays, one per protein
  - accessions:    string array of UniProt accession IDs
  - function_text: string array of function descriptions
  - organism:      string array of organism names

Resume-safe: skips per-chunk npz files that already exist.

Usage:
    conda activate GeoBPE-6
    python scripts/encode_stage2.py --num-workers 56
    python scripts/encode_stage2.py --merge-only
"""

import argparse
import logging
import multiprocessing
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PT-BPE"))

from foldingdiff.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("encode_stage2")

BPE_CKPT = Path("/data/steven/PT-BPE/ckpts/swissprot_michael/bpe_post_init.pkl")
FEAT_DIR = Path("/data/steven/ProteinChamaleon/encoded/featurized")
OUT_DIR  = Path("/data/steven/ProteinChamaleon/encoded")
OUT_FILE = OUT_DIR / "stage2.npz"

# Inherited by forked workers — never set directly in worker processes
_bpe = None


def _chunk_npz_path(pkl_path: Path) -> Path:
    return OUT_DIR / f"stage2_{pkl_path.stem}.npz"


def _sub_npz_path(pkl_path: Path, start: int, end: int) -> Path:
    return OUT_DIR / f"stage2_{pkl_path.stem}_{start}_{end}.npz"


def encode_chunk(bpe, chunk_data: dict) -> dict:
    ids_out, acc_out, func_out, org_out = [], [], [], []
    failed = 0
    rows = zip(
        chunk_data["structures"],
        chunk_data["accessions"],
        chunk_data["function_text"],
        chunk_data["organism"],
    )
    for struct, acc, func, org in tqdm(rows, total=len(chunk_data["accessions"]),
                                       desc="  encoding", leave=False):
        if struct is None:
            failed += 1
            continue
        try:
            struct["angles"] = pd.DataFrame(struct["angles"])
            tok = Tokenizer(struct)
            tok, _ = bpe.tokenize(tok)
            ids = bpe.quantize(tok)
            ids_out.append(np.array(ids, dtype=np.int32))
            acc_out.append(acc)
            func_out.append(func)
            org_out.append(str(org))
        except Exception as e:
            logger.warning("Failed %s: %s", acc, e)
            failed += 1
    return {
        "token_ids":     ids_out,
        "accessions":    acc_out,
        "function_text": func_out,
        "organism":      org_out,
        "failed":        failed,
    }


def save_npz(path: Path, result: dict) -> None:
    id_arr = np.empty(len(result["token_ids"]), dtype=object)
    for i, arr in enumerate(result["token_ids"]):
        id_arr[i] = arr
    np.savez(
        path,
        token_ids=id_arr,
        accessions=np.array(result["accessions"],     dtype=object),
        function_text=np.array(result["function_text"], dtype=object),
        organism=np.array(result["organism"],         dtype=object),
    )


def _get_chunk_size(pkl_path_str: str) -> tuple:
    with open(pkl_path_str, "rb") as f:
        data = pickle.load(f)
    return pkl_path_str, len(data["accessions"])


def _worker(task: tuple) -> tuple:
    """Fork worker: uses _bpe inherited from parent via CoW."""
    pkl_path_str, npz_path_str, start, end = task
    pkl_path = Path(pkl_path_str)
    npz_path = Path(npz_path_str)
    with open(pkl_path, "rb") as f:
        chunk_data = pickle.load(f)
    sub = {k: chunk_data[k][start:end]
           for k in ("structures", "accessions", "function_text", "organism")}
    result = encode_chunk(_bpe, sub)
    save_npz(npz_path, result)
    return len(result["token_ids"]), result["failed"], npz_path.name


def merge(all_chunks: list) -> None:
    all_npz = sorted(OUT_DIR.glob("stage2_chunk_*_*_*.npz"))
    logger.info("Merging %d sub-chunk npz files into %s ...", len(all_npz), OUT_FILE)
    all_ids, all_acc, all_func, all_org = [], [], [], []
    for npz in all_npz:
        data = np.load(npz, allow_pickle=True)
        all_ids.extend(data["token_ids"])
        all_acc.extend(data["accessions"])
        all_func.extend(data["function_text"])
        all_org.extend(data["organism"])

    id_arr = np.empty(len(all_ids), dtype=object)
    for i, arr in enumerate(all_ids):
        id_arr[i] = arr
    np.savez(
        OUT_FILE,
        token_ids=id_arr,
        accessions=np.array(all_acc,   dtype=object),
        function_text=np.array(all_func, dtype=object),
        organism=np.array(all_org,   dtype=object),
    )
    logger.info("Done. Saved %d proteins to %s", len(all_ids), OUT_FILE)


def main(args):
    global _bpe, OUT_DIR, OUT_FILE, FEAT_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = sorted(FEAT_DIR.glob("chunk_*.pkl"))
    if not all_chunks:
        logger.error("No featurized chunks in %s — run featurize_stage2.py first", FEAT_DIR)
        return

    if args.merge_only:
        merge(all_chunks)
        return

    pending = [
        (str(pkl), str(_chunk_npz_path(pkl)))
        for pkl in all_chunks
        if not _chunk_npz_path(pkl).exists()
    ]

    # Scan chunk sizes to build sub-chunk task list (before loading BPE to keep RAM clean)
    logger.info("Scanning %d chunk sizes ...", len(all_chunks))
    with multiprocessing.Pool(min(8, len(all_chunks))) as p:
        sizes = dict(p.map(_get_chunk_size, [str(c) for c in all_chunks]))

    pending = []
    for pkl in all_chunks:
        n = sizes[str(pkl)]
        for start in range(0, n, args.sub_chunk_size):
            end = min(start + args.sub_chunk_size, n)
            npz = _sub_npz_path(pkl, start, end)
            if not npz.exists():
                pending.append((str(pkl), str(npz), start, end))

    if not pending:
        logger.info("All sub-chunks already encoded.")
    else:
        logger.info("%d sub-chunks to encode with %d workers (sub-chunk size=%d)",
                    len(pending), args.num_workers, args.sub_chunk_size)
        logger.info("Loading BPE checkpoint from %s ...", BPE_CKPT)
        with open(BPE_CKPT, "rb") as f:
            _bpe = pickle.load(f)
        logger.info("BPE loaded. Forking workers (CoW — no extra RAM per worker)...")

        # fork inherits _bpe — workers share the 6GB BPE object read-only
        ctx = multiprocessing.get_context("fork")
        total_failed = 0
        with ctx.Pool(processes=args.num_workers) as pool:
            for n_enc, n_fail, name in pool.imap_unordered(_worker, pending):
                total_failed += n_fail
                logger.info("  ✓ %s — %d encoded, %d failed", name, n_enc, n_fail)

        logger.info("Encode complete. Total failed: %d", total_failed)

    if args.merge:
        merge(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers",    type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--sub-chunk-size", type=int, default=500)
    parser.add_argument("--merge-only",     action="store_true")
    parser.add_argument("--merge",       action="store_true", help="merge after encoding")
    parser.add_argument("--bpe-ckpt",    default=str(BPE_CKPT))
    parser.add_argument("--feat-dir",    default=str(FEAT_DIR))
    parser.add_argument("--out-dir",     default=str(OUT_DIR))
    args = parser.parse_args()
    BPE_CKPT = Path(args.bpe_ckpt)
    FEAT_DIR  = Path(args.feat_dir)
    OUT_DIR   = Path(args.out_dir)
    OUT_FILE  = OUT_DIR / "stage2.npz"
    main(args)
