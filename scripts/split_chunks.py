"""
Split each ~1GB chunk_NNNN.pkl into smaller part pkls so encode_stage2.py
workers don't each need to load 1GB+ of memory per task.

Each chunk pkl is a dict with parallel arrays:
    structures, accessions, function_text, organism

We slice each array into PART_SIZE-sized parts and write
    chunk_NNNN_part_MM.pkl
in the same directory. Originals are moved to FEAT_DIR/_originals/ so the
encode_stage2.py glob ("chunk_*.pkl") only picks up the new parts.

Usage:
    python scripts/split_chunks.py --part-size 500 --num-workers 8
"""

import argparse
import multiprocessing as mp
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm


KEYS = ("structures", "accessions", "function_text", "organism")


def split_one(task):
    pkl_path, part_size, feat_dir, backup_dir = task
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    n = len(data["accessions"])
    n_parts = 0
    for part_idx, start in enumerate(range(0, n, part_size)):
        end = min(start + part_size, n)
        part = {k: data[k][start:end] for k in KEYS}
        out_path = feat_dir / f"{pkl_path.stem}_part_{part_idx:02d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_parts += 1

    shutil.move(str(pkl_path), str(backup_dir / pkl_path.name))
    return pkl_path.name, n, n_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path,
                    default=Path.home() / "encoded" / "featurized")
    ap.add_argument("--part-size", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    feat_dir = args.feat_dir
    backup_dir = feat_dir / "_originals"
    backup_dir.mkdir(exist_ok=True)

    chunks = sorted(p for p in feat_dir.glob("chunk_*.pkl") if "_part_" not in p.name)
    if not chunks:
        print(f"No un-split chunks found in {feat_dir}")
        return

    print(f"Splitting {len(chunks)} chunks into ~{args.part_size}-protein parts")
    print(f"  feat_dir   = {feat_dir}")
    print(f"  backup_dir = {backup_dir}")
    print(f"  workers    = {args.num_workers}\n")

    tasks = [(p, args.part_size, feat_dir, backup_dir) for p in chunks]
    total_parts = 0
    total_proteins = 0
    with mp.Pool(args.num_workers) as pool:
        for name, n, n_parts in tqdm(
            pool.imap_unordered(split_one, tasks),
            total=len(tasks),
            desc="splitting",
            unit="chunk",
        ):
            total_parts += n_parts
            total_proteins += n

    print(f"\nDone. {len(chunks)} chunks -> {total_parts} parts "
          f"({total_proteins:,} proteins).")


if __name__ == "__main__":
    main()
