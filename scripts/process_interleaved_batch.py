"""
Step 2: Download GPT-4.1 batch results and build interleaved training examples.

Replaces [IPRxxxxxx] domain placeholders in GPT output with:
  [PROT_START] tok1 tok2 ... [PROT_END]

using the domain GeoBPE tokens from domain_tokens.npz.

Final training example format:
  Organism: <organism>
  Sequence: <full AA sequence>

  <GPT narrative with [PROT_START]...[PROT_END] domain blocks inline>

Output: interleaved_train.npz / interleaved_val.npz / interleaved_test.npz
  accessions   - protein accession
  sequences    - full AA sequence
  organism     - organism name
  narratives   - full interleaved text (with structure tokens as space-separated ints)
  n_domains    - number of domain placeholders successfully replaced

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/process_interleaved_batch.py
    python scripts/process_interleaved_batch.py --batch-id batch_abc123  # override
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI

OUT_DIR          = Path("/data/steven/ProteinChamaleon/encoded/stage2")
DOMAIN_TOKENS_NPZ = OUT_DIR / "domain_tokens.npz"


def get_split(acc: str) -> str:
    h = hash(acc) % 100
    if h < 90: return "train"
    if h < 95: return "val"
    return "test"


def build_domain_token_map(domain_npz_path: Path) -> dict:
    """
    Build {(accession, ipr_acc): token_ids_str} from domain_tokens.npz.
    token_ids_str is space-separated ints ready to insert into text.
    """
    print("Loading domain_tokens.npz...")
    d = np.load(domain_npz_path, allow_pickle=True)
    token_map = {}
    for acc, ipr_acc, tids in zip(d["accessions"], d["domain_acc"], d["token_ids"]):
        key = (str(acc), str(ipr_acc))
        token_map[key] = " ".join(map(str, tids.tolist()))
    print(f"  {len(token_map):,} domain token entries loaded")
    return token_map


def replace_placeholders(text: str, acc: str, token_map: dict) -> tuple[str, int]:
    """
    Replace [IPRxxxxxx] placeholders with [PROT_START] tokens [PROT_END].
    Returns (replaced_text, n_replaced).
    """
    pattern = re.compile(r'\[IPR\d+\]')
    n_replaced = 0

    def replacer(match):
        nonlocal n_replaced
        placeholder = match.group(0)
        ipr_acc = placeholder[1:-1]  # strip brackets → IPRxxxxxx
        key = (acc, ipr_acc)
        if key in token_map:
            n_replaced += 1
            return f"[PROT_START] {token_map[key]} [PROT_END]"
        # placeholder present but no tokens (domain too short, encoding failed)
        return ""

    replaced = pattern.sub(replacer, text)
    # clean up any double spaces from empty replacements
    replaced = re.sub(r'  +', ' ', replaced).strip()
    return replaced, n_replaced


def format_example(organism: str, sequence: str, narrative: str) -> str:
    return f"Organism: {organism}\nSequence: {sequence}\n\n{narrative}"


def main(args):
    # ── Load from cache or download from OpenAI ───────────────────────────────
    raw_cache = args.out_dir / "interleaved_gpt_results.jsonl"
    results   = {}

    if raw_cache.exists() and not args.batch_id:
        print(f"Loading cached GPT results from {raw_cache.name}...")
        with open(raw_cache) as f:
            for line in f:
                obj = json.loads(line)
                results[obj["accession"]] = obj["text"]
        print(f"  {len(results):,} cached results loaded")
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # auto-discover all interleaved_batch_info*.json files
        batch_ids = []
        if args.batch_id:
            batch_ids = [(args.batch_id, "batch")]
        else:
            for info_file in sorted(args.out_dir.glob("interleaved_batch_info*.json")):
                with open(info_file) as f:
                    info = json.load(f)
                batch_ids.append((info["batch_id"], info_file.stem))

        for bid, label in batch_ids:
            print(f"Checking {label}: {bid}...")
            batch = client.batches.retrieve(bid)
            print(f"  Status: {batch.status}  Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
            if batch.status != "completed":
                print(f"  Not complete yet — skipping.")
                continue
            print(f"  Downloading...")
            content = client.files.content(batch.output_file_id).text
            for line in content.strip().split("\n"):
                obj = json.loads(line)
                acc = obj["custom_id"]
                try:
                    text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
                    results[acc] = text
                except Exception:
                    pass

    print(f"\nTotal successful responses: {len(results):,}")

    # ── Load metadata from all meta files ─────────────────────────────────────
    metadata = {}
    for meta_file in sorted(args.out_dir.glob("interleaved_batch_meta*.jsonl")):
        print(f"Loading metadata from {meta_file.name}...")
        with open(meta_file) as f:
            for line in f:
                m = json.loads(line)
                metadata[m["accession"]] = m
    print(f"  {len(metadata):,} metadata entries")

    # ── Save raw GPT results to disk ──────────────────────────────────────────
    raw_path = args.out_dir / "interleaved_gpt_results.jsonl"
    with open(raw_path, "w") as f:
        for acc, text in results.items():
            f.write(json.dumps({"accession": acc, "text": text}) + "\n")
    print(f"Raw GPT results saved → {raw_path}")

    if args.download_only:
        print("--download-only set, stopping here. Run again without flag when domain_tokens.npz is ready.")
        return

    # ── Load domain tokens ─────────────────────────────────────────────────────
    token_map = build_domain_token_map(args.domain_tokens_npz)

    # ── Process each result ────────────────────────────────────────────────────
    splits = {"train": [], "val": [], "test": []}
    skipped_no_meta = skipped_no_domains = 0

    for acc, gpt_text in results.items():
        if acc not in metadata:
            skipped_no_meta += 1
            continue

        meta    = metadata[acc]
        org     = meta["organism"]
        seq     = meta["sequence"]

        replaced, n_domains = replace_placeholders(gpt_text, acc, token_map)

        if n_domains == 0:
            skipped_no_domains += 1
            continue

        example = format_example(org, seq, replaced)
        split   = get_split(acc)
        splits[split].append({
            "accession": acc,
            "sequence":  seq,
            "organism":  org,
            "narrative": example,
            "n_domains": n_domains,
        })

    total = sum(len(v) for v in splits.values())
    print(f"\nProcessed: {total}  |  skipped (no meta): {skipped_no_meta}  |  skipped (no domains replaced): {skipped_no_domains}")

    # ── Print a sample ─────────────────────────────────────────────────────────
    for split_name, records in splits.items():
        if records:
            r = records[0]
            print(f"\n=== Sample ({split_name}) — {r['accession']} — {r['n_domains']} domains replaced ===")
            print(r["narrative"][:600])
            print("...")
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    for split_name, records in splits.items():
        if not records:
            continue
        n       = len(records)
        out_path = args.out_dir / f"interleaved_{split_name}.npz"
        np.savez(
            out_path,
            accessions = np.array([r["accession"] for r in records], dtype=object),
            sequences  = np.array([r["sequence"]  for r in records], dtype=object),
            organism   = np.array([r["organism"]  for r in records], dtype=object),
            narratives = np.array([r["narrative"] for r in records], dtype=object),
            n_domains  = np.array([r["n_domains"] for r in records], dtype=np.int32),
        )
        print(f"Saved {split_name}: {n} examples → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",           default=str(OUT_DIR),           type=Path)
    parser.add_argument("--domain-tokens-npz", default=str(DOMAIN_TOKENS_NPZ), type=Path)
    parser.add_argument("--batch-id",      default=None,
                        help="Override batch ID from interleaved_batch_info.json")
    parser.add_argument("--download-only", action="store_true",
                        help="Download and cache GPT results only, skip domain token replacement")
    args = parser.parse_args()
    main(args)
