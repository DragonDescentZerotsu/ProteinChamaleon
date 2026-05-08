"""
Warmup checkpoint inference — perplexity + generation, with results saved.

Loads a warmup checkpoint (full model — no LoRA), evaluates perplexity on
sample proteins and/or generates new protein token sequences, and writes
all results to an experiment directory:

    <out-dir>/
        run_metadata.json     args, checkpoint info, timestamp
        perplexity.json       per-protein loss/ppl/n_tokens + aggregate
        perplexity.csv        spreadsheet-friendly version of the same
        generated_tokens.npz  generated BPE sequences (warmup.npz format)
        generated_tokens.txt  human-readable token-ID dump

Usage:
    python scripts/infer_warmup.py --mode perplexity --n-samples 20
    python scripts/infer_warmup.py --mode generate   --n-tokens 256 --n-generate 8
    python scripts/infer_warmup.py --mode both
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infer_warmup")

BASE_MODEL    = "/home/ubuntu/models/gemma-4-e4b"
CKPT_DIR      = Path("/home/ubuntu/checkpoints/warmup/checkpoint-400")
ENCODED_FILE  = Path("/home/ubuntu/encoded/warmup.npz")


def load_model(base_model: str, ckpt_dir):
    """Load the model. If ckpt_dir is None or 'none', skip checkpoint loading
    and return the from_gemma-initialized model — useful as a step-0 baseline."""
    logger.info("Loading tokenizer...")
    tokenizer = ProteinChameleonTokenizer.from_pretrained(base_model)

    logger.info("Building base architecture + expanded vocab...")
    model = ProteinChameleonForCausalLM.from_gemma(
        base_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if ckpt_dir is None or str(ckpt_dir).lower() == "none":
        logger.info("No checkpoint specified — using from_gemma init (step-0 baseline).")
        model.eval()
        return model, tokenizer

    ckpt_dir = Path(ckpt_dir)
    shards = sorted(ckpt_dir.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model*.safetensors found in {ckpt_dir}")
    logger.info("Loading %d safetensors shard(s) from %s ...", len(shards), ckpt_dir)
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(str(shard)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint: %d missing keys, %d unexpected keys",
                len(missing), len(unexpected))
    if missing:
        logger.warning("First 5 missing: %s", missing[:5])
    if unexpected:
        logger.warning("First 5 unexpected: %s", unexpected[:5])

    model.eval()
    return model, tokenizer


def build_input(token_ids: np.ndarray, tokenizer, max_length: int, device) -> dict:
    """Build <PROT_START> [shifted protein tokens] <PROT_END> tensor."""
    offset  = tokenizer.protein_token_offset
    shifted = [offset + int(i) for i in token_ids]
    ids     = [tokenizer.prot_start_id] + shifted + [tokenizer.prot_end_id]
    ids     = ids[:max_length]
    t       = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask    = torch.ones_like(t)
    return {"input_ids": t, "attention_mask": mask}


@torch.inference_mode()
def eval_perplexity(model, tokenizer, token_ids_list, source_indices, max_length=1024):
    """Compute per-protein loss/ppl. Returns list of dicts and aggregate dict."""
    device = next(model.parameters()).device
    prot_lo = tokenizer.protein_token_offset
    prot_hi = prot_lo + tokenizer.protein_vocab_size

    records = []
    for src_idx, raw_ids in zip(source_indices, token_ids_list):
        inp = build_input(raw_ids, tokenizer, max_length, device)
        out = model(**inp)

        shift_logits = out.logits[0, :-1, prot_lo:prot_hi].float()
        shift_labels = inp["input_ids"][0, 1:]

        is_struct = (shift_labels >= prot_lo) & (shift_labels < prot_hi)
        local_labels = shift_labels.clone()
        local_labels[is_struct]  -= prot_lo
        local_labels[~is_struct] = -100

        loss = F.cross_entropy(shift_logits, local_labels, ignore_index=-100)
        ppl  = torch.exp(loss).item()
        n_toks = int(is_struct.sum().item())
        rec = {
            "source_idx":  int(src_idx),
            "raw_length":  int(len(raw_ids)),
            "ctx_length":  int(inp["input_ids"].shape[1]),
            "n_struct_tokens": n_toks,
            "loss":        float(loss.item()),
            "perplexity":  float(ppl),
        }
        records.append(rec)
        logger.info("  src=%d  loss=%.4f  ppl=%.2f  (%d protein tokens)",
                    src_idx, rec["loss"], rec["perplexity"], n_toks)

    losses = [r["loss"] for r in records]
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    aggregate = {
        "n_proteins":  len(records),
        "mean_loss":   avg_loss,
        "median_loss": float(np.median(losses)) if losses else float("nan"),
        "std_loss":    float(np.std(losses)) if losses else float("nan"),
        "min_loss":    float(np.min(losses)) if losses else float("nan"),
        "max_loss":    float(np.max(losses)) if losses else float("nan"),
        "mean_ppl":    float(np.exp(avg_loss)) if losses else float("nan"),
    }
    logger.info("── Aggregate: mean_loss=%.4f  mean_ppl=%.2f over %d proteins",
                aggregate["mean_loss"], aggregate["mean_ppl"], aggregate["n_proteins"])
    return records, aggregate


@torch.inference_mode()
def generate_one(model, tokenizer, n_tokens, temperature, top_k):
    device = next(model.parameters()).device
    prot_lo = tokenizer.protein_token_offset
    prot_hi = prot_lo + tokenizer.protein_vocab_size

    ids = torch.tensor([[tokenizer.prot_start_id]], dtype=torch.long, device=device)
    generated = []
    stopped_at_end = False

    for _ in range(n_tokens):
        out = model(input_ids=ids)
        logits = out.logits[0, -1, prot_lo:prot_hi].float()

        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[-1]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        tok   = torch.multinomial(probs, 1).item()
        unified_id = prot_lo + tok
        generated.append(tok)

        ids = torch.cat([ids, torch.tensor([[unified_id]], device=device)], dim=1)

        if unified_id == tokenizer.prot_end_id:
            stopped_at_end = True
            break

    return generated, stopped_at_end


def generate(model, tokenizer, n_tokens=256, temperature=1.0, top_k=50, n_generate=1):
    """Run autoregressive generation n_generate times. Returns list of dicts."""
    samples = []
    logger.info("Generating %d sample(s) of up to %d tokens (temp=%.2f, top_k=%d)...",
                n_generate, n_tokens, temperature, top_k)
    for i in range(n_generate):
        toks, stopped = generate_one(model, tokenizer, n_tokens, temperature, top_k)
        rec = {
            "sample_idx":      i,
            "n_tokens":        len(toks),
            "stopped_at_end":  bool(stopped),
            "tokens":          toks,
        }
        samples.append(rec)
        preview = toks[:30]
        logger.info("  sample %d: len=%d  stopped_at_end=%s  preview=%s%s",
                    i, len(toks), stopped, preview,
                    " ..." if len(toks) > len(preview) else "")
    return samples


def resolve_out_dir(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser()
    if args.ckpt_dir is None or str(args.ckpt_dir).lower() == "none":
        ckpt_name = "checkpoint-0"
    else:
        ckpt_name = Path(args.ckpt_dir).name or "ckpt"
    return Path("/home/ubuntu/eval_results/warmup") / ckpt_name


def write_perplexity_csv(records, csv_path: Path) -> None:
    if not records:
        return
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


def write_generated_outputs(samples, npz_path: Path, txt_path: Path,
                            base_model: str, ckpt_dir: str, gen_args: dict) -> None:
    # NPZ in warmup.npz format: object array of int32 arrays (+ a names array)
    token_arrays = np.array(
        [np.asarray(s["tokens"], dtype=np.int32) for s in samples],
        dtype=object,
    )
    fnames = np.array([f"generated_sample_{s['sample_idx']:04d}" for s in samples], dtype=object)
    np.savez(npz_path, token_ids=token_arrays, fnames=fnames)

    with open(txt_path, "w") as f:
        f.write(f"# Generated protein-token samples\n")
        f.write(f"# base_model: {base_model}\n")
        f.write(f"# ckpt_dir:   {ckpt_dir}\n")
        f.write(f"# gen_args:   {json.dumps(gen_args)}\n")
        f.write(f"# n_samples:  {len(samples)}\n\n")
        for s in samples:
            f.write(f"## sample_idx={s['sample_idx']}  n_tokens={s['n_tokens']}  "
                    f"stopped_at_end={s['stopped_at_end']}\n")
            f.write(", ".join(str(t) for t in s["tokens"]))
            f.write("\n\n")


def main(args):
    out_dir = resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s", out_dir)

    model, tokenizer = load_model(args.base_model, Path(args.ckpt_dir))

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model":    args.base_model,
        "ckpt_dir":      str(Path(args.ckpt_dir).resolve()),
        "encoded_file":  str(Path(args.encoded_file).resolve())
                         if Path(args.encoded_file).exists() else args.encoded_file,
        "mode":          args.mode,
        "args":          vars(args),
        "torch":         torch.__version__,
        "device":        str(next(model.parameters()).device),
    }

    if args.mode in ("perplexity", "both"):
        logger.info("Loading encoded proteins from %s", args.encoded_file)
        data = np.load(args.encoded_file, allow_pickle=True)
        token_ids = data["token_ids"]

        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(token_ids), size=min(args.n_samples, len(token_ids)), replace=False)
        samples = [token_ids[i] for i in idxs]
        logger.info("Evaluating perplexity on %d proteins (seed=%d)...", len(samples), args.seed)
        records, aggregate = eval_perplexity(
            model, tokenizer, samples, source_indices=idxs, max_length=args.max_length,
        )

        ppl_payload = {
            "checkpoint":   metadata["ckpt_dir"],
            "encoded_file": metadata["encoded_file"],
            "max_length":   args.max_length,
            "seed":         args.seed,
            "aggregate":    aggregate,
            "per_protein":  records,
        }
        with open(out_dir / "perplexity.json", "w") as f:
            json.dump(ppl_payload, f, indent=2)
        write_perplexity_csv(records, out_dir / "perplexity.csv")
        logger.info("Saved perplexity.json + perplexity.csv  (n=%d, mean_ppl=%.2f)",
                    aggregate["n_proteins"], aggregate["mean_ppl"])

    if args.mode in ("generate", "both"):
        gen_args = {
            "n_tokens":    args.n_tokens,
            "temperature": args.temperature,
            "top_k":       args.top_k,
            "n_generate":  args.n_generate,
        }
        gen_samples = generate(model, tokenizer, **gen_args)
        write_generated_outputs(
            gen_samples,
            npz_path=out_dir / "generated_tokens.npz",
            txt_path=out_dir / "generated_tokens.txt",
            base_model=args.base_model,
            ckpt_dir=metadata["ckpt_dir"],
            gen_args=gen_args,
        )
        logger.info("Saved generated_tokens.npz + generated_tokens.txt  (n=%d)", len(gen_samples))

    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved run_metadata.json")
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["perplexity", "generate", "both"], default="both")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--ckpt-dir",   default=str(CKPT_DIR))
    parser.add_argument("--encoded-file", default=str(ENCODED_FILE))
    parser.add_argument("--out-dir",    default=None,
                        help="Where to write results. Defaults to "
                             "/home/ubuntu/eval_results/warmup/<ckpt-name>/")
    parser.add_argument("--n-samples",  type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--n-tokens",   type=int, default=256)
    parser.add_argument("--n-generate", type=int, default=1,
                        help="How many independent generated sequences to sample.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k",      type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    main(args)
