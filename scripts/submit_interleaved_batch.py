"""
Step 1: Prepare and submit GPT-4.1 batch requests for interleaved dataset.

For each eligible protein, builds a prompt containing:
  - protein name, organism, function_text
  - full AA sequence
  - domain list with name, type, residue range, and domain AA subsequence

Asks GPT-4.1 to write a scientific narrative that interleaves domain
structure references as [DOMAIN:IPRxxxxxx] placeholders.

Outputs:
  interleaved_batch_input.jsonl  — uploaded to OpenAI Batch API
  interleaved_batch_meta.jsonl   — custom_id → protein metadata (for post-processing)

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/submit_interleaved_batch.py --max-proteins 10   # test
    python scripts/submit_interleaved_batch.py                      # full run
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

PROTEINS_CSV = Path("/home/steven/ProteinChamaleon-Dataset/output/proteins.csv")
FEATURES_CSV = Path("/home/steven/ProteinChamaleon-Dataset/output/features.csv")
ELIGIBLE_TXT = Path("/tmp/eligible_accs.txt")
OUT_DIR      = Path("/data/steven/ProteinChamaleon/encoded/stage2")

DOMAIN_TYPES = {"Domain"}
MIN_DOMAINS  = 2

SYSTEM_PROMPT = (
    "You are a biomedical scientist writing training data for a multimodal protein AI model. "
    "Write factual, concise scientific prose about the given protein. "
    "Do not invent facts. If evidence is limited, say so briefly. "
    "Do not use bullet points, numbered lists, or section headers."
)


def build_user_prompt(name: str, organism: str, sequence: str,
                      function_text: str, domains: list[dict]) -> str:
    domain_lines = "\n".join(
        f"  - [{d['ipr_acc']}] {d['ipr_name']} "
        f"(residues {d['start']}–{d['end']}, {d['end']-d['start']+1} aa): "
        f"{d['domain_seq']}"
        for d in domains
    )

    return (
        f"Protein: {name}\n"
        f"Organism: {organism}\n"
        f"Full sequence ({len(sequence)} aa): {sequence}\n\n"
        f"Known function:\n{function_text}\n\n"
        f"Structural domains (use at least 2 placeholders):\n{domain_lines}\n\n"
        "Task: Write a 4–6 sentence scientific narrative describing this protein's "
        "structure and function. Where you discuss a domain, insert its placeholder "
        "exactly as shown (e.g. [IPR000198]) immediately after naming it. "
        "Use each domain placeholder at most once. "
        "Write in coherent prose — no lists, no headers.\n\n"
        "Output format rules:\n"
        "- Use placeholders in the form [IPRxxxxxx] — the accession only, no brackets around the name.\n"
        "- Only use placeholders from the domain list above.\n"
        "- Every placeholder must appear inline within a sentence, not at the start or end.\n"
        "- Cover: overall function, how specific domains contribute, any known biological context."
    )


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading proteins.csv...")
    proteins = pd.read_csv(args.proteins_csv,
                           usecols=["accession", "name", "sequence"])
    proteins = proteins.set_index("accession")

    print("Loading features.csv...")
    feat_df = pd.read_csv(args.features_csv)
    feat_df = feat_df[feat_df["ipr_type"].isin(DOMAIN_TYPES)]

    print("Loading stage2.npz for function_text + organism...")
    s2 = np.load(args.out_dir / "encoded" / "stage2.npz", allow_pickle=True)
    DENYLIST = {"uncharacterized protein", "hypothetical protein",
                "unknown function", "function unknown"}
    acc_to_func = dict(zip(s2["accessions"], s2["function_text"]))
    acc_to_org  = dict(zip(s2["accessions"], s2["organism"]))

    print("Loading eligible accessions...")
    with open(args.eligible_txt) as f:
        eligible = set(l.strip() for l in f if l.strip())

    # filter to proteins that are in alignment train + have >= min_domains
    counts     = feat_df[feat_df["protein_acc"].isin(eligible)].groupby("protein_acc").size()
    valid_accs = list(counts[counts >= args.min_domains].index)

    # further filter: must have function_text >= 300 chars in stage2
    valid_accs = [a for a in valid_accs
                  if a in acc_to_func
                  and len(str(acc_to_func[a])) >= 300
                  and not any(x in str(acc_to_func[a]).lower() for x in DENYLIST)]

    print(f"Eligible proteins: {len(valid_accs):,}")

    if args.offset:
        valid_accs = valid_accs[args.offset:]
        print(f"Offset by {args.offset}, remaining: {len(valid_accs):,}")
    if args.max_proteins:
        valid_accs = valid_accs[:args.max_proteins]
        print(f"Capped to: {len(valid_accs)}")

    # ── Build JSONL ───────────────────────────────────────────────────────────
    suffix = f"_{args.suffix}" if args.suffix else ""
    batch_input_path = args.out_dir / f"interleaved_batch_input{suffix}.jsonl"
    meta_path        = args.out_dir / f"interleaved_batch_meta{suffix}.jsonl"

    skipped = 0
    written = 0

    with open(batch_input_path, "w") as fb, open(meta_path, "w") as fm:
        for acc in valid_accs:
            if acc not in proteins.index:
                skipped += 1
                continue

            row      = proteins.loc[acc]
            sequence = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
            name     = str(row["name"]) if pd.notna(row["name"]) else acc
            func     = acc_to_func[acc]
            org      = acc_to_org.get(acc, "Unknown")

            if not sequence:
                skipped += 1
                continue

            # get domains for this protein
            domains_df = feat_df[feat_df["protein_acc"] == acc]
            domains = []
            for _, d in domains_df.iterrows():
                s, e = int(d["start"]), int(d["end"])
                domain_seq = sequence[s-1:e] if e <= len(sequence) else sequence[s-1:]
                domains.append({
                    "ipr_acc":   d["ipr_acc"],
                    "ipr_name":  d["ipr_name"],
                    "start":     s,
                    "end":       e,
                    "domain_seq": domain_seq,
                })

            if len(domains) < args.min_domains:
                skipped += 1
                continue

            user_prompt = build_user_prompt(name, org, sequence, func, domains)

            # OpenAI Batch API request format
            request = {
                "custom_id": acc,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":       "gpt-4.1",
                    "max_tokens":  600,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
            }
            fb.write(json.dumps(request) + "\n")

            # metadata for post-processing
            meta = {
                "accession": acc,
                "organism":  org,
                "sequence":  sequence,
                "domains":   domains,
            }
            fm.write(json.dumps(meta) + "\n")
            written += 1

    print(f"Written: {written} requests  |  Skipped: {skipped}")

    # ── Check file size — warn if approaching 200MB OpenAI limit ─────────────
    size_mb = batch_input_path.stat().st_size / 1e6
    print(f"JSONL size: {size_mb:.1f} MB")
    if size_mb > 190:
        print(f"WARNING: file is {size_mb:.1f} MB — close to 200MB OpenAI limit.")
        print("Consider splitting with --max-proteins to run in multiple batches.")
        return

    print(f"Batch input: {batch_input_path}")
    print(f"Metadata:    {meta_path}")

    # ── Upload + submit batch ──────────────────────────────────────────────────
    print("\nUploading batch file to OpenAI...")
    with open(batch_input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file id: {uploaded.id}")

    print("Submitting batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "ProteinChameleon interleaved dataset"},
    )
    print(f"Batch id:     {batch.id}")
    print(f"Status:       {batch.status}")

    # save batch id for later retrieval
    info_path = args.out_dir / f"interleaved_batch_info{suffix}.json"
    with open(info_path, "w") as f:
        json.dump({"batch_id": batch.id, "file_id": uploaded.id,
                   "n_requests": written}, f, indent=2)
    print(f"Batch info saved → {info_path}")
    print("\nRun process_interleaved_batch.py once the batch completes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins-csv",  default=str(PROTEINS_CSV), type=Path)
    parser.add_argument("--features-csv",  default=str(FEATURES_CSV), type=Path)
    parser.add_argument("--eligible-txt",  default=str(ELIGIBLE_TXT), type=Path)
    parser.add_argument("--out-dir",       default=str(OUT_DIR),      type=Path)
    parser.add_argument("--min-domains",   default=MIN_DOMAINS,       type=int)
    parser.add_argument("--max-proteins",  default=None,              type=int)
    parser.add_argument("--offset",        default=0,                 type=int,
                        help="Skip first N proteins (for splitting into multiple batches)")
    parser.add_argument("--suffix",        default="",
                        help="Suffix for output files, e.g. 'part1' -> interleaved_batch_input_part1.jsonl")
    args = parser.parse_args()
    main(args)
