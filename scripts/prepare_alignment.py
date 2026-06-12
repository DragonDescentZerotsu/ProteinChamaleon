"""
Prepare the Stage II alignment dataset from stage2.npz + proteins.csv.

Filters:
  - function_text length >= 150 chars, no boilerplate
  - structure token sequence length <= MAX_TOKENS (default 4096)
  - AA sequence available in proteins.csv

Split 90/5/5 train/val/test by accession hash (deterministic, homolog-safe).

Output npz keys: token_ids, accessions, sequences, function_text, organism

Usage:
    python scripts/prepare_alignment.py
    python scripts/prepare_alignment.py --max-tokens 2048
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_alignment")

STAGE2_NPZ   = Path("/data/steven/ProteinChamaleon/encoded/stage2/encoded/stage2.npz")
PROTEINS_CSV = Path("/home/steven/ProteinChamaleon-Dataset/output/proteins.csv")
OUT_DIR      = Path("/data/steven/ProteinChamaleon/encoded/stage2")
MAX_TOKENS   = 4096
MIN_FUNC_LEN = 150
DENYLIST     = {
    "uncharacterized protein", "hypothetical protein", "unknown function",
    "function unknown", "not yet known", "predicted protein",
}


def get_split(acc: str) -> str:
    h = hash(acc) % 100
    if h < 90:  return "train"
    if h < 95:  return "val"
    return "test"


def is_good_func(text: str, min_len: int) -> bool:
    if not text or len(text) < min_len:
        return False
    return not any(d in text.lower() for d in DENYLIST)


def save_split(records: list, name: str, out_dir: Path) -> None:
    n = len(records)
    id_arr   = np.empty(n, dtype=object)
    acc_arr  = np.empty(n, dtype=object)
    seq_arr  = np.empty(n, dtype=object)
    func_arr = np.empty(n, dtype=object)
    org_arr  = np.empty(n, dtype=object)
    for i, r in enumerate(records):
        id_arr[i]   = r["token_ids"]
        acc_arr[i]  = r["accession"]
        seq_arr[i]  = r["sequence"]
        func_arr[i] = r["function_text"]
        org_arr[i]  = r["organism"]
    path = out_dir / f"alignment_{name}.npz"
    np.savez(path, token_ids=id_arr, accessions=acc_arr,
             sequences=seq_arr, function_text=func_arr, organism=org_arr)
    logger.info("  %-6s  %7d proteins → %s", name, n, path)


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading sequence map from proteins.csv ...")
    seq_map = (pd.read_csv(args.proteins_csv, usecols=["accession", "sequence"])
               .set_index("accession")["sequence"].to_dict())
    logger.info("  %d sequences loaded", len(seq_map))

    logger.info("Loading stage2.npz ...")
    d = np.load(args.stage2_npz, allow_pickle=True)
    token_ids     = d["token_ids"]
    accessions    = d["accessions"]
    function_text = d["function_text"]
    organisms     = d["organism"]
    logger.info("  %d proteins loaded", len(token_ids))

    splits = {"train": [], "val": [], "test": []}
    skipped = {"func": 0, "len": 0, "seq": 0}

    for i in tqdm(range(len(token_ids)), desc="filtering", unit="prot"):
        acc  = str(accessions[i])
        func = str(function_text[i]) if function_text[i] is not None else ""
        ids  = token_ids[i]

        if not is_good_func(func, args.min_func_len):
            skipped["func"] += 1
            continue
        if len(ids) > args.max_tokens:
            skipped["len"] += 1
            continue
        seq = seq_map.get(acc)
        if not seq:
            skipped["seq"] += 1
            continue

        splits[get_split(acc)].append({
            "token_ids":    ids,
            "accession":    acc,
            "sequence":     seq,
            "function_text": func,
            "organism":     str(organisms[i]) if organisms[i] is not None else "",
        })

    total_kept = sum(len(v) for v in splits.values())
    logger.info(
        "Kept %d / %d proteins  (skipped: func=%d, len=%d, seq=%d)",
        total_kept, len(token_ids),
        skipped["func"], skipped["len"], skipped["seq"],
    )

    logger.info("Saving splits:")
    for name, records in splits.items():
        save_split(records, name, args.out_dir)

    logger.info(
        "Done — train: %d | val: %d | test: %d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-npz",   default=str(STAGE2_NPZ),   type=Path)
    parser.add_argument("--proteins-csv", default=str(PROTEINS_CSV), type=Path)
    parser.add_argument("--out-dir",      default=str(OUT_DIR),      type=Path)
    parser.add_argument("--max-tokens",   default=MAX_TOKENS,        type=int)
    parser.add_argument("--min-func-len", default=MIN_FUNC_LEN,      type=int)
    args = parser.parse_args()
    main(args)
