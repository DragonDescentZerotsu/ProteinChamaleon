"""
Encode InterPro domain fragments for the interleaved dataset.

Loads BPE checkpoint ONCE in the main process, then forks workers that share
it via copy-on-write (same pattern as encode_stage2.py). Total RAM is ~6GB
(shared BPE) + N x ~1GB (chunk data) instead of N x 7GB.

For each eligible protein, slices featurized angle arrays by InterPro domain
residue ranges and BPE-encodes each slice.

Output: domain_tokens.npz
  accessions   - protein UniProt accession
  domain_acc   - InterPro accession (IPRxxxxxx)
  domain_name  - InterPro entry name
  domain_type  - Domain / Active_site / Binding_site / Conserved_site
  start        - domain start residue (1-indexed)
  end          - domain end residue (1-indexed)
  token_ids    - GeoBPE token IDs for the domain fragment
  split        - train / val / test (same hash split as alignment)

Usage:
    /data/steven/miniconda3/envs/GeoBPE-6/bin/python scripts/prepare_domains.py
    /data/steven/miniconda3/envs/GeoBPE-6/bin/python scripts/prepare_domains.py --num-workers 32 --max-proteins 100
"""

import argparse
import logging
import multiprocessing
import os
import pickle
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.captureWarnings(True)
for _noisy in ("transformers", "torch", "foldingdiff", "esm", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PT-BPE"))
from foldingdiff.tokenizer import Tokenizer

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("prepare_domains")
logger.setLevel(logging.INFO)

# ── Defaults (override with CLI args for node001) ─────────────────────────────
BPE_CKPT     = Path("/data/steven/PT-BPE/ckpts/swissprot_michael/bpe_post_init.pkl")
FEAT_DIR     = Path("/data/steven/ProteinChamaleon/encoded/stage2/featurized")
FEATURES_CSV = Path("/home/steven/ProteinChamaleon-Dataset/output/features.csv")
ELIGIBLE_TXT = Path("/tmp/eligible_accs.txt")   # pre-generated list of accessions
OUT_DIR      = Path("/data/steven/ProteinChamaleon/encoded/stage2")

DOMAIN_TYPES   = {"Domain"}
MIN_DOMAINS    = 2
MIN_DOMAIN_LEN = 10  # skip fragments shorter than 10 residues

# Inherited by forked workers via CoW
_bpe = None


def get_split(acc: str) -> str:
    h = hash(acc) % 100
    if h < 90: return "train"
    if h < 95: return "val"
    return "test"


def _worker_init():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _encode_protein(task: tuple) -> list:
    """
    Fork worker. Uses _bpe inherited from parent via CoW.
    task = (acc, chunk_path_str, row_idx, domains_list)
    domains_list = [(ipr_acc, ipr_name, ipr_type, start, end), ...]
    Returns list of result dicts.
    """
    acc, chunk_path_str, row_idx, domains = task
    results = []

    try:
        with open(chunk_path_str, "rb") as f:
            chunk = pickle.load(f)
        struct = chunk["structures"][row_idx]
        del chunk
    except Exception:
        return results

    if struct is None:
        return results

    angles = struct["angles"]
    n_res  = len(next(iter(angles.values())))
    split  = get_split(acc)

    for ipr_acc, ipr_name, ipr_type, start, end in domains:
        s0 = max(0, start - 1)
        s1 = min(end, n_res)
        if s1 - s0 < MIN_DOMAIN_LEN:
            continue
        try:
            sliced = dict(struct)
            sliced["angles"] = pd.DataFrame({k: v[s0:s1] for k, v in angles.items()})
            tok = Tokenizer(sliced)
            tok, _ = _bpe.tokenize(tok, compute_metrics=False)
            ids = np.array(_bpe.quantize(tok), dtype=np.int32)
            results.append({
                "accession":   acc,
                "domain_acc":  ipr_acc,
                "domain_name": ipr_name,
                "domain_type": ipr_type,
                "start":       start,
                "end":         end,
                "token_ids":   ids,
                "split":       split,
            })
        except Exception:
            continue

    return results


def build_accession_index(feat_dir: Path) -> dict:
    """Build {accession: (chunk_path_str, row_idx)} from all featurized chunks."""
    tqdm.write("  Building accession index...")
    index = {}
    for chunk_path in tqdm(sorted(feat_dir.glob("chunk_*.pkl")),
                           desc="  indexing", unit="chunk"):
        with open(chunk_path, "rb") as f:
            chunk = pickle.load(f)
        for row_idx, acc in enumerate(chunk["accessions"]):
            index[acc] = (str(chunk_path), row_idx)
    tqdm.write(f"  {len(index):,} proteins indexed.")
    return index


def main(args):
    global _bpe
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"  ProteinChameleon — Domain Fragment Encoding")
    tqdm.write(f"{'─'*60}")
    tqdm.write(f"  Workers  : {args.num_workers}")
    tqdm.write(f"  Feat dir : {args.feat_dir}")
    tqdm.write(f"  Out dir  : {args.out_dir}")
    tqdm.write(f"{'─'*60}\n")

    # ── Load eligible accessions ───────────────────────────────────────────────
    tqdm.write("  Loading eligible accessions...")
    with open(args.eligible_txt) as f:
        eligible = set(l.strip() for l in f if l.strip())
    tqdm.write(f"  {len(eligible):,} eligible proteins")

    # ── Load InterPro features ─────────────────────────────────────────────────
    tqdm.write("  Loading InterPro features...")
    feat_df = pd.read_csv(args.features_csv)
    feat_df = feat_df[feat_df["ipr_type"].isin(DOMAIN_TYPES)]
    feat_df = feat_df[feat_df["protein_acc"].isin(eligible)]

    counts     = feat_df.groupby("protein_acc").size()
    valid_accs = set(counts[counts >= args.min_domains].index)
    feat_df    = feat_df[feat_df["protein_acc"].isin(valid_accs)]
    tqdm.write(f"  {len(valid_accs):,} proteins with >= {args.min_domains} domain features")
    tqdm.write(f"  {len(feat_df):,} total domain fragments")

    if args.max_proteins:
        valid_accs = set(list(sorted(valid_accs))[:args.max_proteins])
        feat_df    = feat_df[feat_df["protein_acc"].isin(valid_accs)]
        tqdm.write(f"  Capped to {len(valid_accs):,} proteins")

    # ── Build featurized chunk index ───────────────────────────────────────────
    acc_index = build_accession_index(args.feat_dir)

    # ── Build task list ────────────────────────────────────────────────────────
    # Group domains by protein, build one task per protein
    grouped = feat_df.groupby("protein_acc")
    tasks = []
    for acc in sorted(valid_accs):
        if acc not in acc_index:
            continue
        chunk_path_str, row_idx = acc_index[acc]
        domains = [
            (row["ipr_acc"], row["ipr_name"], row["ipr_type"],
             int(row["start"]), int(row["end"]))
            for _, row in grouped.get_group(acc).iterrows()
            if acc in grouped.groups
        ]
        if domains:
            tasks.append((acc, chunk_path_str, row_idx, domains))

    tqdm.write(f"  {len(tasks):,} proteins to process\n")

    # ── Load BPE ──────────────────────────────────────────────────────────────
    tqdm.write("  Loading BPE checkpoint...")
    with open(args.bpe_ckpt, "rb") as f:
        _bpe = pickle.load(f)
    tqdm.write(f"  BPE loaded → forking {args.num_workers} workers\n")

    # ── Encode ────────────────────────────────────────────────────────────────
    all_results = []
    ctx = multiprocessing.get_context("fork")

    with ctx.Pool(processes=args.num_workers,
                  initializer=_worker_init) as pool:
        for protein_results in tqdm(
            pool.imap_unordered(_encode_protein, tasks, chunksize=4),
            total=len(tasks),
            desc="  encoding",
            unit="protein",
        ):
            all_results.extend(protein_results)

    tqdm.write(f"\n  Encoded {len(all_results):,} domain fragments total")

    # ── Save ──────────────────────────────────────────────────────────────────
    n      = len(all_results)
    id_arr = np.empty(n, dtype=object)
    for i, r in enumerate(all_results):
        id_arr[i] = r["token_ids"]

    out_path = args.out_dir / "domain_tokens.npz"
    np.savez(
        out_path,
        accessions  = np.array([r["accession"]   for r in all_results], dtype=object),
        domain_acc  = np.array([r["domain_acc"]  for r in all_results], dtype=object),
        domain_name = np.array([r["domain_name"] for r in all_results], dtype=object),
        domain_type = np.array([r["domain_type"] for r in all_results], dtype=object),
        start       = np.array([r["start"]       for r in all_results], dtype=np.int32),
        end         = np.array([r["end"]         for r in all_results], dtype=np.int32),
        token_ids   = id_arr,
        split       = np.array([r["split"]       for r in all_results], dtype=object),
    )
    tqdm.write(f"  Saved → {out_path}")

    splits = {}
    for r in all_results:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    tqdm.write(f"  Split: {splits}")
    tqdm.write(f"\n{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe-ckpt",     default=str(BPE_CKPT),     type=Path)
    parser.add_argument("--feat-dir",     default=str(FEAT_DIR),     type=Path)
    parser.add_argument("--features-csv", default=str(FEATURES_CSV), type=Path)
    parser.add_argument("--eligible-txt", default=str(ELIGIBLE_TXT), type=Path)
    parser.add_argument("--out-dir",      default=str(OUT_DIR),      type=Path)
    parser.add_argument("--min-domains",  default=MIN_DOMAINS,       type=int)
    parser.add_argument("--num-workers",  default=multiprocessing.cpu_count(), type=int)
    parser.add_argument("--max-proteins", default=None,              type=int)
    args = parser.parse_args()
    main(args)
