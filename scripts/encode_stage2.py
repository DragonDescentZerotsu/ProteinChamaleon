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
import shutil
import sys
import threading
import time
import warnings
from collections import defaultdict
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
# Set in worker by _worker_init
_protein_counter = None

HB_DIR = Path("/tmp/encode_stage2_heartbeats")
HB_INTERVAL = 5      # write a heartbeat every N proteins (and once at start/end)
STALL_SECONDS = 30   # worker is "stalled" if no heartbeat update in this long


def _chunk_npz_path(pkl_path: Path) -> Path:
    return OUT_DIR / f"stage2_{pkl_path.stem}.npz"


def _sub_npz_path(pkl_path: Path, start: int, end: int) -> Path:
    return OUT_DIR / f"stage2_{pkl_path.stem}_{start}_{end}.npz"


def _worker_init(protein_counter):
    """Silence noisy loggers, store shared protein counter, ensure HB dir."""
    global _protein_counter
    _protein_counter = protein_counter
    HB_DIR.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def encode_chunk(bpe, chunk_data: dict, heartbeat_fn=None) -> dict:
    ids_out, acc_out, func_out, org_out = [], [], [], []
    failed = 0
    rows = list(zip(
        chunk_data["structures"],
        chunk_data["accessions"],
        chunk_data["function_text"],
        chunk_data["organism"],
    ))
    n = len(rows)
    for i, (struct, acc, func, org) in enumerate(rows):
        if struct is None:
            failed += 1
        else:
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
        if heartbeat_fn is not None and (i + 1) % HB_INTERVAL == 0:
            heartbeat_fn(i + 1, n)
    if heartbeat_fn is not None:
        heartbeat_fn(n, n)
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


def _parent_chunk(pkl_stem: str) -> tuple:
    """('chunk_0001_part_03') -> ('chunk_0001', 'part_03')
       ('chunk_0001')         -> ('chunk_0001', '')
    """
    if "_part_" in pkl_stem:
        parent, part_suffix = pkl_stem.rsplit("_part_", 1)
        return parent, f"part_{part_suffix}"
    return pkl_stem, ""


def _worker(task: tuple) -> tuple:
    """Fork worker: uses _bpe inherited from parent via CoW."""
    pkl_path_str, npz_path_str, start, end = task
    pkl_path = Path(pkl_path_str)
    npz_path = Path(npz_path_str)
    pid = os.getpid()
    hb_path = HB_DIR / f"hb_{pid}.txt"

    last_reported = 0
    def _hb(i, n):
        nonlocal last_reported
        delta = i - last_reported
        if _protein_counter is not None and delta > 0:
            with _protein_counter.get_lock():
                _protein_counter.value += delta
        last_reported = i
        try:
            hb_path.write_text(f"{time.time():.2f}\t{pkl_path.stem}\t{i}\t{n}\n")
        except OSError:
            pass

    # Mark task START (i=0) so monitor can spot stalls before any protein finishes
    try:
        hb_path.write_text(f"{time.time():.2f}\t{pkl_path.stem}\t0\t-1\n")
    except OSError:
        pass

    with open(pkl_path_str, "rb") as f:
        chunk_data = pickle.load(f)
    sub = {k: chunk_data[k][start:end]
           for k in ("structures", "accessions", "function_text", "organism")}
    result = encode_chunk(_bpe, sub, heartbeat_fn=_hb)
    save_npz(npz_path, result)
    return len(result["token_ids"]), result["failed"], pkl_path.stem, npz_path.name


def _heartbeat_monitor(counter, t_start, n_proteins_total, stop_event,
                        interval: float = 10.0):
    """Daemon thread: every `interval` sec, print global protein progress
    plus list any workers whose heartbeat is older than STALL_SECONDS."""
    last_count = 0
    last_time = t_start
    while not stop_event.wait(interval):
        now_time = time.time()
        try:
            now_count = counter.value
        except Exception:
            now_count = last_count
        period   = max(now_time - last_time, 1e-3)
        delta    = now_count - last_count
        rate     = delta / period
        avg_rate = now_count / max(now_time - t_start, 1e-3)
        remaining = max(n_proteins_total - now_count, 0)
        eta_s = remaining / max(avg_rate, 1e-3) if avg_rate > 0 else float("inf")
        if eta_s == float("inf"):
            eta_str = "?"
        elif eta_s < 3600:
            eta_str = f"{eta_s/60:.1f}min"
        else:
            eta_str = f"{eta_s/3600:.1f}h"

        active, stalled, stalls_detail = 0, 0, []
        if HB_DIR.exists():
            for hb in HB_DIR.glob("hb_*.txt"):
                try:
                    txt = hb.read_text().strip()
                    if not txt:
                        continue
                    parts = txt.split("\t")
                    ts = float(parts[0])
                    age = now_time - ts
                    if age > STALL_SECONDS:
                        stalled += 1
                        if len(stalls_detail) < 5:
                            chunk_name = parts[1] if len(parts) > 1 else "?"
                            i = parts[2] if len(parts) > 2 else "?"
                            n = parts[3] if len(parts) > 3 else "?"
                            pid = hb.stem.removeprefix("hb_")
                            stalls_detail.append(
                                f"pid={pid} {chunk_name} stuck at {i}/{n} "
                                f"({age:.0f}s ago)"
                            )
                    else:
                        active += 1
                except Exception:
                    pass

        pct = 100 * now_count / n_proteins_total if n_proteins_total else 0
        tqdm.write(
            f"  [hb] proteins {now_count:,}/{n_proteins_total:,} ({pct:.1f}%)  "
            f"+{delta:,} in {period:.0f}s = {rate:.0f}/s  "
            f"avg {avg_rate:.0f}/s  ETA {eta_str}  "
            f"[active={active}, stalled={stalled}]"
        )
        for s in stalls_detail:
            tqdm.write(f"      ⚠ STALLED: {s}")

        last_count = now_count
        last_time  = now_time


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

        # ── Per-parent-chunk progress tracking ────────────────────────────
        parent_progress = defaultdict(
            lambda: {"done": 0, "total": 0, "encoded": 0, "failed": 0}
        )
        for pkl_str, _, _, _ in pending:
            parent, _ = _parent_chunk(Path(pkl_str).stem)
            parent_progress[parent]["total"] += 1
        n_parents_total = len(parent_progress)
        n_parents_done  = 0

        # ── Encode ────────────────────────────────────────────────────────
        ctx = multiprocessing.get_context("fork")
        total_failed = 0
        t_start = time.time()

        # Fresh heartbeat dir
        if HB_DIR.exists():
            shutil.rmtree(HB_DIR)
        HB_DIR.mkdir(parents=True, exist_ok=True)

        # Shared counter for proteins encoded across all workers
        protein_counter = ctx.Value("Q", 0)
        n_proteins_pending = sum(end - start for _, _, start, end in pending)

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
        bar.set_postfix(chunks=f"0/{n_parents_total}",
                        enc=0, failed=0, refresh=False)

        # Daemon thread: aggregate per-protein heartbeats from workers
        hb_stop = threading.Event()
        hb_thread = threading.Thread(
            target=_heartbeat_monitor,
            args=(protein_counter, t_start, n_proteins_pending, hb_stop),
            daemon=True,
        )
        hb_thread.start()

        with ctx.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(protein_counter,),
        ) as pool:
            n_enc_total = 0
            for n_enc, n_fail, pkl_stem, npz_name in pool.imap_unordered(_worker, pending):
                parent, part = _parent_chunk(pkl_stem)
                state = parent_progress[parent]
                state["done"]    += 1
                state["encoded"] += n_enc
                state["failed"]  += n_fail
                n_enc_total  += n_enc
                total_failed += n_fail

                # Per-task line with parent context
                part_label = part if part else "(whole)"
                fail_str = f"  ✗ {n_fail} failed" if n_fail else ""
                tqdm.write(
                    f"  ✓ {parent:<14} {part_label:<10} "
                    f"[{state['done']:>3}/{state['total']:<3}]  "
                    f"{n_enc:>4} enc"
                    f"{fail_str}  "
                    f"({state['encoded']:>6,} so far in {parent})"
                )

                # Per-chunk completion announcement
                if state["done"] == state["total"]:
                    n_parents_done += 1
                    fail_part = (f", {state['failed']} failed"
                                 if state['failed'] else "")
                    tqdm.write(
                        f"  ✅ {parent} COMPLETE  "
                        f"{state['encoded']:>6,} encoded{fail_part}  "
                        f"[{n_parents_done}/{n_parents_total} chunks done]"
                    )

                # Live aggregates in postfix
                elapsed = max(time.time() - t_start, 1e-3)
                rate = n_enc_total / elapsed
                bar.set_postfix(
                    chunks=f"{n_parents_done}/{n_parents_total}",
                    enc=f"{n_enc_total:,}",
                    failed=total_failed,
                    rate=f"{rate:.0f}/s",
                    refresh=False,
                )
                bar.update(1)

        hb_stop.set()
        hb_thread.join(timeout=2)
        bar.close()
        elapsed = time.time() - t_start
        rate = n_enc_total / max(elapsed, 1e-3)
        _print(f"\n{'─'*60}")
        _print(f"  Encoding complete")
        _print(f"  Encoded : {n_enc_total:,}")
        _print(f"  Failed  : {total_failed:,}")
        _print(f"  Chunks  : {n_parents_done}/{n_parents_total}")
        _print(f"  Elapsed : {elapsed:.0f}s  ({rate:.0f} proteins/s)")
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
