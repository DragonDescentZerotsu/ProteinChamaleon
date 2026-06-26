"""
Stage II evaluation script for ProteinChameleonForCausalLM.

Computes:
  1. Perplexity on alignment test set
  2. Perplexity on interleaved test set
  3. Generation quality on alignment test set
       - Prompt:  [BOS] Organism: ...\nSequence: ...\n<PROT_START> struct_tokens <PROT_END>
       - Generate: function text until EOS
       - Metrics: BLEU-4, ROUGE-1/2/L vs ground-truth function text

Usage:
    python scripts/eval_stage2.py \
        --ckpt checkpoints/stage2/final \
        --align-test  /home/steven/data/stage2/alignment/alignment_test.npz \
        --interleaved-test /home/steven/data/stage2/interleaved_test.npz \
        --n-gen 500 \
        --out-dir eval_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM
from scripts.train_stage2 import (
    AlignmentDataset,
    InterleavedDataset,
    encode_alignment,
    PackingCollator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_stage2")


# ── Perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(
    dataset,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    desc: str,
) -> float:
    collator = PackingCollator(pad_id=tokenizer.pad_id, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids      = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            loss_mask      = batch["loss_mask"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask   = loss_mask[:, 1:].contiguous()

            labels = shift_labels.clone()
            labels[shift_mask == 0] = -100

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (labels != -100).sum().item()
            total_loss   += loss.item()
            total_tokens += n_valid

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    logger.info("%s  loss=%.4f  perplexity=%.4f", desc, avg_loss, ppl)
    return avg_loss, ppl


# ── Generation ────────────────────────────────────────────────────────────────

def build_prompt(organism, sequence, struct_bpe_ids, tokenizer) -> list[int]:
    """Encode everything up to (but not including) the function text."""
    offset = tokenizer.protein_token_offset
    prefix_text = f"Organism: {organism}\nSequence: {sequence}\n"
    ids = (
        [tokenizer.text_tokenizer.bos_token_id]
        + tokenizer.encode_text(prefix_text)
        + [tokenizer.prot_start_id]
        + [offset + i for i in struct_bpe_ids]
        + [tokenizer.prot_end_id]
    )
    return ids


def decode_text_only(token_ids: list[int], tokenizer) -> str:
    """Decode only text tokens, skipping protein structure tokens."""
    text_ids = [
        t for t in token_ids
        if t < tokenizer.protein_token_offset
        and t != tokenizer.prot_start_id
        and t != tokenizer.prot_end_id
        and t != tokenizer.eos_id
        and t != tokenizer.text_tokenizer.bos_token_id
    ]
    return tokenizer.text_tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def run_generation(
    npz_path: Path,
    model,
    tokenizer,
    n_examples: int,
    max_new_tokens: int,
    max_prompt_length: int,
    out_dir: Path,
    shard: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    d = np.load(npz_path, allow_pickle=True)
    token_ids_all  = d["token_ids"]
    sequences      = d["sequences"]
    function_texts = d["function_text"]
    organisms      = d["organism"]

    results = []
    indices = np.random.default_rng(42).choice(len(token_ids_all), size=min(n_examples, len(token_ids_all)), replace=False)
    indices = indices[shard::num_shards]  # this shard's slice

    model.eval()
    for idx in tqdm(indices, desc="Generating"):
        accession = str(d["accessions"][idx])
        example_dir = out_dir / accession
        json_path   = example_dir / f"{accession}.json"

        # Resume: skip if already done
        if json_path.exists():
            results.append(json.loads(json_path.read_text()))
            continue

        prompt_ids = build_prompt(
            organism      = str(organisms[idx]),
            sequence      = str(sequences[idx]),
            struct_bpe_ids= token_ids_all[idx].tolist(),
            tokenizer     = tokenizer,
        )

        if len(prompt_ids) > max_prompt_length:
            continue

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens   = max_new_tokens,
                do_sample        = False,
                eos_token_id     = tokenizer.eos_id,
                pad_token_id     = tokenizer.pad_id,
            )

        generated_ids  = output[0][len(prompt_ids):].tolist()
        generated_text = decode_text_only(generated_ids, tokenizer)
        ground_truth   = str(function_texts[idx])

        record = {
            "organism"      : str(organisms[idx]),
            "accession"     : accession,
            "sequence"      : str(sequences[idx]),
            "ground_truth"  : ground_truth,
            "generated"     : generated_text,
            "struct_bpe_ids": token_ids_all[idx].tolist(),
        }

        example_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(record, indent=2))
        results.append(record)

    return results


# ── Interleaved generation ────────────────────────────────────────────────────

def run_interleaved_generation(
    npz_path: Path,
    model,
    tokenizer,
    n_examples: int,
    max_new_tokens: int,
    max_prompt_length: int,
    out_dir: Path,
    shard: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    d = np.load(npz_path, allow_pickle=True)
    narratives = d["narratives"]
    organisms  = d["organism"]
    sequences  = d["sequences"]
    accessions = d["accessions"]

    results = []
    indices = np.random.default_rng(42).choice(len(narratives), size=min(n_examples, len(narratives)), replace=False)
    indices = indices[shard::num_shards]  # this shard's slice

    model.eval()
    for idx in tqdm(indices, desc="Generating (interleaved)"):
        accession   = str(accessions[idx])
        example_dir = out_dir / accession
        json_path   = example_dir / f"{accession}.json"

        if json_path.exists():
            results.append(json.loads(json_path.read_text()))
            continue

        # Prompt: just organism + sequence header, model generates the narrative
        prompt_text = f"Organism: {organisms[idx]}\nSequence: {sequences[idx]}\n\n"
        prompt_ids = (
            [tokenizer.text_tokenizer.bos_token_id]
            + tokenizer.encode_text(prompt_text)
        )

        if len(prompt_ids) > max_prompt_length:
            continue

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                eos_token_id   = tokenizer.eos_id,
                pad_token_id   = tokenizer.pad_id,
            )

        generated_ids = output[0][len(prompt_ids):].tolist()
        offset = tokenizer.protein_token_offset

        # Split generated_ids into segments: {"type": "text"/"structure", "content": ...}
        # text segments → decoded string, structure segments → list of BPE token IDs
        segments = []
        prot_start = tokenizer.prot_start_id
        prot_end   = tokenizer.prot_end_id
        i = 0
        while i < len(generated_ids):
            if generated_ids[i] == prot_start:
                # collect until prot_end or end of sequence
                i += 1
                bpe_ids = []
                while i < len(generated_ids) and generated_ids[i] != prot_end:
                    tok = generated_ids[i]
                    if tok >= offset:
                        bpe_ids.append(tok - offset)
                    i += 1
                i += 1  # skip prot_end
                if bpe_ids:
                    segments.append({"type": "structure", "bpe_ids": bpe_ids})
            else:
                # collect text tokens until next prot_start or end
                text_ids = []
                while i < len(generated_ids) and generated_ids[i] != prot_start:
                    text_ids.append(generated_ids[i])
                    i += 1
                text = tokenizer.text_tokenizer.decode(text_ids, skip_special_tokens=True)
                if text.strip():
                    segments.append({"type": "text", "content": text})

        ground_truth = str(narratives[idx])
        header = prompt_text.strip()
        gt_body = ground_truth[ground_truth.find(header) + len(header):].strip() if header in ground_truth else ground_truth

        # Parse GT narrative into segments using inline [PROT_START] ... [PROT_END] markers
        import re as _re
        gt_segments = []
        gt_pattern = _re.compile(r'\[PROT_START\](.*?)\[PROT_END\]', _re.DOTALL)
        last_end = 0
        for m in gt_pattern.finditer(gt_body):
            text_before = gt_body[last_end:m.start()].strip()
            if text_before:
                gt_segments.append({"type": "text", "content": text_before})
            bpe_ids = [int(x) for x in m.group(1).split() if x.strip().isdigit()]
            if bpe_ids:
                gt_segments.append({"type": "structure", "bpe_ids": bpe_ids})
            last_end = m.end()
        text_after = gt_body[last_end:].strip()
        if text_after:
            gt_segments.append({"type": "text", "content": text_after})

        record = {
            "organism"      : str(organisms[idx]),
            "accession"     : accession,
            "sequence"      : str(sequences[idx]),
            "gt_segments"   : gt_segments,
            "segments"      : segments,
        }

        example_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(record, indent=2))
        results.append(record)

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for r in results:
            s = scorer.score(r["ground_truth"], r["generated"])
            for k in rouge_scores:
                rouge_scores[k].append(s[k].fmeasure)
        rouge_summary = {k: float(np.mean(v)) for k, v in rouge_scores.items()}
    except ImportError:
        logger.warning("rouge_score not installed — skipping ROUGE. pip install rouge-score")
        rouge_summary = {}

    try:
        import sacrebleu
        refs  = [[r["ground_truth"] for r in results]]
        hyps  = [r["generated"]     for r in results]
        bleu  = sacrebleu.corpus_bleu(hyps, refs)
        bleu_score = {"bleu4": bleu.score}
    except ImportError:
        logger.warning("sacrebleu not installed — skipping BLEU. pip install sacrebleu")
        bleu_score = {}

    try:
        from bert_score import score as bert_score_fn
        hyps = [r["generated"]    for r in results]
        refs = [r["ground_truth"] for r in results]
        P, R, F1 = bert_score_fn(hyps, refs, lang="en", verbose=False)
        bert_scores = {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall":    float(R.mean()),
            "bertscore_f1":        float(F1.mean()),
        }
    except ImportError:
        logger.warning("bert-score not installed — skipping. pip install bert-score")
        bert_scores = {}

    return {**rouge_summary, **bleu_score, **bert_scores}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.ckpt)
    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.ckpt)
    model = ProteinChameleonForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results_summary = {}

    if not args.gen_only:
        # ── Alignment perplexity ──────────────────────────────────────────────
        align_test = AlignmentDataset(Path(args.align_test), tokenizer, args.max_length)
        loss, ppl = compute_perplexity(
            align_test, model, tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            desc="Alignment perplexity",
        )
        results_summary["alignment_loss"] = loss
        results_summary["alignment_ppl"]  = ppl

        # ── Interleaved perplexity ────────────────────────────────────────────
        inter_test = InterleavedDataset(Path(args.interleaved_test), tokenizer, args.max_length)
        loss, ppl = compute_perplexity(
            inter_test, model, tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            desc="Interleaved perplexity",
        )
        results_summary["interleaved_loss"] = loss
        results_summary["interleaved_ppl"]  = ppl

    # ── Generation + BLEU/ROUGE ───────────────────────────────────────────────
    logger.info("Running generation on %d alignment test examples", args.n_gen)
    gen_results = run_generation(
        npz_path         = Path(args.align_test),
        model            = model,
        tokenizer        = tokenizer,
        n_examples       = args.n_gen,
        max_new_tokens   = args.max_new_tokens,
        max_prompt_length= args.max_length - args.max_new_tokens,
        out_dir          = out_dir / "alignment",
        shard            = args.shard,
        num_shards       = args.num_shards,
    )

    if gen_results:
        metrics = compute_metrics(gen_results)
        results_summary["generation_metrics"] = metrics
        logger.info("Generation metrics: %s", metrics)
    results_summary["n_generated"] = len(gen_results)

    # ── Interleaved generation ────────────────────────────────────────────────
    if args.n_gen_interleaved > 0:
        logger.info("Running interleaved generation on %d examples", args.n_gen_interleaved)
        inter_gen_results = run_interleaved_generation(
            npz_path         = Path(args.interleaved_test),
            model            = model,
            tokenizer        = tokenizer,
            n_examples       = args.n_gen_interleaved,
            max_new_tokens   = args.max_new_tokens_interleaved,
            max_prompt_length= args.max_length - args.max_new_tokens_interleaved,
            out_dir          = out_dir / "interleaved",
            shard            = args.shard,
            num_shards       = args.num_shards,
        )
        logger.info("Interleaved generation saved to %s/interleaved/", out_dir)

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    examples_path = out_dir / "generation_examples.jsonl"
    with open(examples_path, "w") as f:
        for r in gen_results:
            f.write(json.dumps(r) + "\n")
    logger.info("Generation examples saved to %s", examples_path)

    print("\n=== Results ===")
    if "alignment_loss" in results_summary:
        print(f"Alignment   loss={results_summary['alignment_loss']:.4f}  ppl={results_summary['alignment_ppl']:.4f}")
    if "interleaved_loss" in results_summary:
        print(f"Interleaved loss={results_summary['interleaved_loss']:.4f}  ppl={results_summary['interleaved_ppl']:.4f}")
    if gen_results and "generation_metrics" in results_summary:
        for k, v in results_summary["generation_metrics"].items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",              default="checkpoints/stage2/final")
    parser.add_argument("--align-test",        default="/data2/steven/data/stage2/alignment/alignment_test_clean.npz")
    parser.add_argument("--interleaved-test",  default="/data2/steven/data/stage2/interleaved_test_clean.npz")
    parser.add_argument("--out-dir",           default="eval_results")
    parser.add_argument("--max-length",        type=int, default=8192)
    parser.add_argument("--batch-size",        type=int, default=1)
    parser.add_argument("--n-gen",             type=int, default=500,
                        help="Number of alignment examples to run generation on")
    parser.add_argument("--max-new-tokens",    type=int, default=256,
                        help="Max tokens to generate for alignment function text")
    parser.add_argument("--max-new-tokens-interleaved", type=int, default=2048,
                        help="Max tokens to generate for interleaved narratives")
    parser.add_argument("--gen-only",          action="store_true",
                        help="Skip perplexity eval, only run generation")
    parser.add_argument("--n-gen-interleaved", type=int, default=0,
                        help="Number of interleaved examples to generate (0 = skip)")
    parser.add_argument("--shard",             type=int, default=0,
                        help="Which shard to process (0-indexed)")
    parser.add_argument("--num-shards",        type=int, default=1,
                        help="Total number of shards (for multi-GPU parallelism)")
    args = parser.parse_args()
    main(args)
