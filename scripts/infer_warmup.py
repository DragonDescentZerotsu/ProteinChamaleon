"""
Warmup checkpoint inference — perplexity + generation.

Loads checkpoint-1000 (LoRA adapter) on top of the base Gemma4 model,
then either evaluates perplexity on sample proteins or generates new
protein token sequences autoregressively.

Usage:
    python scripts/infer_warmup.py --mode perplexity --n-samples 20
    python scripts/infer_warmup.py --mode generate   --n-tokens 256
    python scripts/infer_warmup.py --mode both
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infer_warmup")

BASE_MODEL    = "google/gemma-4-E4B"
CKPT_DIR      = Path("/data/steven/ProteinChamaleon/checkpoints/warmup/checkpoint-1000")
ENCODED_FILE  = Path("/data/steven/ProteinChamaleon/encoded/warmup.npz")


def load_model(base_model: str, ckpt_dir: Path):
    logger.info("Loading tokenizer...")
    tokenizer = ProteinChameleonTokenizer.from_pretrained(base_model)

    logger.info("Loading base model + expanding vocab...")
    base = ProteinChameleonForCausalLM.from_gemma(
        base_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading LoRA adapter from %s ...", ckpt_dir)
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
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
def eval_perplexity(model, tokenizer, token_ids_list, max_length=1024):
    device = next(model.parameters()).device
    prot_lo = tokenizer.protein_token_offset
    prot_hi = prot_lo + tokenizer.protein_vocab_size

    losses = []
    for raw_ids in token_ids_list:
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
        n_toks = is_struct.sum().item()
        losses.append((loss.item(), ppl, n_toks))
        logger.info("  loss=%.4f  ppl=%.2f  (%d protein tokens)", loss.item(), ppl, n_toks)

    avg_loss = np.mean([l for l, _, _ in losses])
    avg_ppl  = np.exp(avg_loss)
    logger.info("── Average: loss=%.4f  ppl=%.2f over %d proteins", avg_loss, avg_ppl, len(losses))
    return avg_ppl


@torch.inference_mode()
def generate(model, tokenizer, n_tokens=256, temperature=1.0, top_k=50):
    device = next(model.parameters()).device
    prot_lo = tokenizer.protein_token_offset
    prot_hi = prot_lo + tokenizer.protein_vocab_size

    # Start with <PROT_START>
    ids = torch.tensor([[tokenizer.prot_start_id]], dtype=torch.long, device=device)
    generated = []

    logger.info("Generating %d protein tokens (temp=%.2f, top_k=%d)...", n_tokens, temperature, top_k)
    for step in range(n_tokens):
        out = model(input_ids=ids)
        logits = out.logits[0, -1, prot_lo:prot_hi].float()

        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[-1]] = -float("inf")

        probs  = torch.softmax(logits, dim=-1)
        tok    = torch.multinomial(probs, 1).item()
        unified_id = prot_lo + tok
        generated.append(tok)

        ids = torch.cat([ids, torch.tensor([[unified_id]], device=device)], dim=1)

        # Stop at PROT_END if it appears
        if unified_id == tokenizer.prot_end_id:
            break

    logger.info("Generated %d raw BPE token IDs:", len(generated))
    logger.info("%s", generated[:50], )
    if len(generated) > 50:
        logger.info("  ... (truncated, full length=%d)", len(generated))
    return generated


def main(args):
    model, tokenizer = load_model(args.base_model, Path(args.ckpt_dir))

    if args.mode in ("perplexity", "both"):
        logger.info("Loading encoded proteins from %s", args.encoded_file)
        data = np.load(args.encoded_file, allow_pickle=True)
        token_ids = data["token_ids"]

        rng = np.random.default_rng(42)
        idxs = rng.choice(len(token_ids), size=min(args.n_samples, len(token_ids)), replace=False)
        samples = [token_ids[i] for i in idxs]
        logger.info("Evaluating perplexity on %d proteins...", len(samples))
        eval_perplexity(model, tokenizer, samples, max_length=args.max_length)

    if args.mode in ("generate", "both"):
        tokens = generate(model, tokenizer, n_tokens=args.n_tokens,
                          temperature=args.temperature, top_k=args.top_k)
        print("\nGenerated BPE token IDs:")
        print(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["perplexity", "generate", "both"], default="both")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--ckpt-dir",   default=str(CKPT_DIR))
    parser.add_argument("--encoded-file", default=str(ENCODED_FILE))
    parser.add_argument("--n-samples",  type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--n-tokens",   type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k",      type=int, default=50)
    args = parser.parse_args()
    main(args)
