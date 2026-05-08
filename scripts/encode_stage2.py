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
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

# ── Silence everything noisy before any imports ────────────────────────────
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.captureWarnings(True)
for _noisy in ("transformers", "torch", "foldingdiff", "esm", "urllib3",
               "filelock", "huggingface_hub", "py.warnings"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PT-BPE"))

from foldingdiff.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.WARNING,          # only our logger prints below WARNING
    format="%(message)s",
)
logger = logging.getLogger("encode_stage2")
logger.setLevel(logging.INFO)

BPE_CKPT = Path("/data/steven/PT-BPE/ckpts/swissprot_michael/bpe_post_init.pkl")
FEAT_DIR = Path("/data/steven/ProteinChamaleon/encoded/featurized")
OUT_DIR  = Path("/data/steven/ProteinChamaleon/encoded")
OUT_FILE = OUT_DIR / "stage2.npz"

# Inherited by forked workers via CoW — never assigned in workers
_bpe = None


def _chunk_npz_path(pkl_path: Path) -> Path:
    return OUT_DIR / f"stage2_{pkl_path.stem}.npz"


def _sub_npz_path(pkl_path: Path, start: int, end: int) -> Path:
    return OUT_DIR / f"stage2_{pkl_path.stem}_{start}_{end}.npz"


def _worker_init():
    """Silence all warnings/logging in forked workers."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def encode_chunk(bpe, chunk_data: dict) -> dict:
    ids_out, acc_out, func_out, org_out = [], [], [], []
    failed = 0
    rows = zip(
        chunk_data["structures"],
        chunk_data["accessions"],
        chunk_data["function_text"],
        chunk_data["organism"],
    )
    for struct, acc, func, org in rows:
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
        except Exception:
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
    npz_path = Path(npz_path_str)
    with open(pkl_path_str, "rb") as f:
        chunk_data = pickle.load(f)
    sub = {k: chunk_data[k][start:end]
           for k in ("structures", "accessions", "function_text", "organism")}
    result = encode_chunk(_bpe, sub)
    save_npz(npz_path, result)
    return len(result["token_ids"]), result["failed"], npz_path.name


def _print(msg: str) -> None:
    tqdm.write(msg)


def merge(all_chunks: list) -> None:
    all_npz = sorted(OUT_DIR.glob("stage2_chunk_*_*_*.npz"))
    _print(f"\n  Merging {len(all_npz)} sub-chunk files → {OUT_FILE.name}")
    all_ids, all_acc, all_func, all_org = [], [], [], []
    for npz in tqdm(all_npz, desc="  merging", unit="file",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
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
    _print(f"\n  Saved {len(all_ids):,} proteins → {OUT_FILE}")


def main(args):
    global _bpe, OUT_DIR, OUT_FILE, FEAT_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = sorted(FEAT_DIR.glob("chunk_*.pkl"))
    if not all_chunks:
        _print(f"ERROR: no featurized chunks in {FEAT_DIR} — run featurize_stage2.py first")
        return

    if args.merge_only:
        merge(all_chunks)
        return

    # ── Scan chunk sizes ───────────────────────────────────────────────────
    _print(f"\n{'─'*60}")
    _print(f"  ProteinChameleon Stage 2 — Encoding")
    _print(f"{'─'*60}")
    _print(f"  Chunks found : {len(all_chunks)}")
    _print(f"  Workers      : {args.num_workers}")
    _print(f"  Sub-chunk sz : {args.sub_chunk_size} proteins/task")
    _print(f"{'─'*60}\n")

    _print("  Scanning chunk sizes...")
    with multiprocessing.Pool(min(8, len(all_chunks))) as p:
        sizes = dict(
            tqdm(
                p.imap_unordered(_get_chunk_size, [str(c) for c in all_chunks]),
                total=len(all_chunks),
                desc="  scanning",
                unit="chunk",
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            )
        )

    # ── Build task list ────────────────────────────────────────────────────
    pending = []
    for pkl in all_chunks:
        n = sizes[str(pkl)]
        for start in range(0, n, args.sub_chunk_size):
            end = min(start + args.sub_chunk_size, n)
            npz = _sub_npz_path(pkl, start, end)
            if not npz.exists():
                pending.append((str(pkl), str(npz), start, end))

    total_tasks   = sum(
        len(range(0, sizes[str(p)], args.sub_chunk_size)) for p in all_chunks
    )
    already_done  = total_tasks - len(pending)
    total_proteins = sum(sizes.values())

    _print(f"\n  Tasks total  : {total_tasks:,}  ({already_done:,} already done, {len(pending):,} pending)")
    _print(f"  Proteins     : {total_proteins:,} total\n")

    if not pending:
        _print("  All sub-chunks already encoded.")
    else:
        # ── Load BPE ──────────────────────────────────────────────────────
        _print(f"  Loading BPE checkpoint...")
        t0 = time.time()
        with open(BPE_CKPT, "rb") as f:
            _bpe = pickle.load(f)
        _print(f"  BPE loaded in {time.time()-t0:.1f}s  →  forking {args.num_workers} workers\n")

        # ── Encode ────────────────────────────────────────────────────────
        ctx = multiprocessing.get_context("fork")
        total_encoded = already_done * args.sub_chunk_size   # rough starting estimate
        total_failed  = 0

        bar = tqdm(
            total=len(pending),
            desc="  encoding",
            unit="task",
            dynamic_ncols=True,
            bar_format=(
                "  {l_bar}{bar}| {n_fmt}/{total_fmt} tasks"
                "  [{elapsed}<{remaining}, {rate_fmt}]"
                "  {postfix}"
            ),
        )
        bar.set_postfix(encoded=0, failed=0, refresh=False)

        with ctx.Pool(processes=args.num_workers, initializer=_worker_init) as pool:
            n_enc_total = 0
            for n_enc, n_fail, name in pool.imap_unordered(_worker, pending):
                n_enc_total  += n_enc
                total_failed += n_fail
                bar.set_postfix(encoded=f"{n_enc_total:,}", failed=total_failed, refresh=False)
                bar.update(1)
                # one-line completion note above the bar
                fail_str = f"  {n_fail} failed" if n_fail else ""
                tqdm.write(f"  ✓  {name:<45}  {n_enc:>4} encoded{fail_str}")

        bar.close()
        _print(f"\n{'─'*60}")
        _print(f"  Encoding complete")
        _print(f"  Encoded : {n_enc_total:,}")
        _print(f"  Failed  : {total_failed:,}")
        _print(f"{'─'*60}\n")

    if args.merge:
        merge(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers",    type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--sub-chunk-size", type=int, default=500)
    parser.add_argument("--merge-only",     action="store_true")
    parser.add_argument("--merge",          action="store_true", help="merge after encoding")
    parser.add_argument("--bpe-ckpt",       default=str(BPE_CKPT))
    parser.add_argument("--feat-dir",       default=str(FEAT_DIR))
    parser.add_argument("--out-dir",        default=str(OUT_DIR))
    args = parser.parse_args()
    BPE_CKPT = Path(args.bpe_ckpt)
    FEAT_DIR  = Path(args.feat_dir)
    OUT_DIR   = Path(args.out_dir)
    OUT_FILE  = OUT_DIR / "stage2.npz"
    main(args)
