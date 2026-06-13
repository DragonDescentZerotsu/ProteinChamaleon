"""
Stage 1 — Protein token warmup training.

Trains ProteinChameleonForCausalLM on protein-only token sequences using
next-token prediction. Stage 1's only job is to give the new vocab rows
(PROT_START, PROT_END, ~2100 structure tokens) reasonable representations,
so all of Gemma is frozen and only the new rows of embed_tokens / lm_head
are updated. Stage 2 handles modality integration (LoRA / full FT there).

Input:  warmup.npz (pre-encoded PT-BPE token arrays)
Output: checkpoints/warmup/

Usage:
    python scripts/train_warmup.py --base-model google/gemma-4-E4B
    python scripts/train_warmup.py --base-model google/gemma-4-E4B \
        --encoded-file /path/to/warmup.npz \
        --out-dir /path/to/checkpoints/warmup \
        --max-length 1024 --batch-size 4 --grad-accum 8 --steps 1000
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("warmup")


# ── Dataset ───────────────────────────────────────────────────────────────────

class WarmupTrainer(Trainer):
    def __init__(self, *args, protein_token_offset: int, protein_vocab_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.protein_token_offset = protein_token_offset
        self.protein_vocab_size   = protein_vocab_size
        # Force Trainer's legacy loss-normalization path: divide by grad_accum
        # once inside training_step. Our compute_loss returns plain per-batch
        # mean CE; without this flag, modern Trainer takes the num_items_in_batch
        # path and over-reports train loss by exactly grad_accum.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Slice out only the 2100 protein structure token logits
        prot_lo = self.protein_token_offset
        prot_hi = self.protein_token_offset + self.protein_vocab_size
        protein_logits = shift_logits[:, :, prot_lo:prot_hi]   # [B, T-1, 2100]


        # Remap labels to [0, protein_vocab_size-1]; mask all non-structure positions
        is_struct = (shift_labels >= prot_lo) & (shift_labels < prot_hi)
        protein_labels = shift_labels.clone()
        protein_labels[is_struct]  -= prot_lo
        protein_labels[~is_struct] = -100

        loss = F.cross_entropy(
            protein_logits.reshape(-1, self.protein_vocab_size),
            protein_labels.reshape(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


class ProteinCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        max_len = max(x.size(0) for x in input_ids)
        batch_ids  = torch.full((len(input_ids), max_len), self.pad_id, dtype=torch.long)
        attn_mask  = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            batch_ids[i, :ids.size(0)] = ids
            attn_mask[i, :ids.size(0)] = 1
        labels = batch_ids.clone()
        labels[attn_mask == 0] = -100
        return {"input_ids": batch_ids, "attention_mask": attn_mask, "labels": labels}


class WarmupDataset(Dataset):
    def __init__(self, token_ids, tokenizer, max_length=1024):
        self.token_ids  = token_ids
        self.prot_start = tokenizer.prot_start_id
        self.prot_end   = tokenizer.prot_end_id
        self.offset     = tokenizer.protein_token_offset
        self.max_length = max_length

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        shifted = [self.offset + i for i in self.token_ids[idx].tolist()]
        ids = ([self.prot_start] + shifted + [self.prot_end])[:self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading encoded proteins from %s", args.encoded_file)
    data = np.load(args.encoded_file, allow_pickle=True)
    token_ids = data["token_ids"]
    logger.info("Loaded %d proteins", len(token_ids))

    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.base_model)

    indices = list(range(len(token_ids)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * 0.05))
    train_dataset = Subset(WarmupDataset(token_ids, tokenizer, args.max_length), indices[n_val:])
    val_dataset   = Subset(WarmupDataset(token_ids, tokenizer, args.max_length), indices[:n_val])
    logger.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading %s", args.base_model)
    # Load the bulk of Gemma in bf16 to keep memory tractable; we'll selectively
    # cast just the trainable embedding matrix to fp32 below so AdamW has stable
    # master weights / optimizer states for the only params we actually update.
    model = ProteinChameleonForCausalLM.from_gemma(
        args.base_model,
        tokenizer=tokenizer,
        use_qk_norm=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Cast the trainable embedding (and tied lm_head) to fp32. With bf16 leaf
    # params, AdamW stores moments in bf16 and the second moment underflows
    # within ~2 steps → NaN gradients. fp32 here costs only ~2-4 GB extra and
    # is well worth it.
    emb_module = model.get_input_embeddings()
    emb_module.weight.data = emb_module.weight.data.float()
    lm_head_module = model.get_output_embeddings()
    if lm_head_module.weight.data_ptr() != emb_module.weight.data_ptr():
        lm_head_module.weight.data = lm_head_module.weight.data.float()

    # Initialize new vocab rows (PROT_START, PROT_END, structure tokens) from
    # the empirical distribution of the original embedding rows. Leaving them
    # at zero makes the first step's grad_norm spike (here: 57 at step 1).
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        n_orig_rows = tokenizer.prot_start_id
        orig_mean = emb[:n_orig_rows].mean(dim=0)
        orig_std  = emb[:n_orig_rows].std(dim=0).mean().item()
        emb[n_orig_rows:].normal_(mean=0.0, std=orig_std * 0.1)
        emb[n_orig_rows:] += orig_mean
        # Gemma ties lm_head to embed_tokens, but if untied for any reason,
        # mirror the init so logits aren't degenerate.
        lm_head_w = model.get_output_embeddings().weight
        if lm_head_w.data_ptr() != emb.data_ptr():
            lm_head_w[n_orig_rows:].copy_(emb[n_orig_rows:])

    # Freeze everything; selectively unfreeze the embedding/lm_head matrices
    # and zero the gradient rows for the original Gemma vocab so only the new
    # rows (PROT_START, PROT_END, structure tokens) actually move.
    orig_vocab = tokenizer.prot_start_id

    def _make_protein_only_hook(n_orig):
        def _hook(grad):
            grad = grad.clone()
            grad[:n_orig] = 0
            return grad
        return _hook

    for param in model.parameters():
        param.requires_grad_(False)

    trainable_params = []
    seen = set()
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            if id(param) in seen:
                continue
            seen.add(id(param))
            param.requires_grad_(True)
            param.register_hook(_make_protein_only_hook(orig_vocab))
            trainable_params.append((name, param))

    n_train = sum(p.numel() for _, p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable: %d params across %d tensors  (%.4f%% of %d total). "
        "Grad-row mask zeros indices [0, %d); only rows [%d:] update.",
        n_train, len(trainable_params), 100 * n_train / n_total, n_total,
        orig_vocab, orig_vocab,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    collator = ProteinCollator(pad_id=tokenizer.pad_id)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=200,
        logging_steps=5,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=5.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="warmup-stage1",
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = WarmupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        protein_token_offset=tokenizer.protein_token_offset,
        protein_vocab_size=tokenizer.protein_vocab_size,
    )

    logger.info("Starting warmup training for %d steps", args.steps)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.text_tokenizer.save_pretrained(str(out_dir / "final"))
    logger.info("Done. Saved to %s", out_dir / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",    default="google/gemma-4-E4B")
    parser.add_argument("--encoded-file",  default="encoded/warmup.npz")
    parser.add_argument("--out-dir",       default="checkpoints/warmup")
    parser.add_argument("--max-length",    type=int, default=1024)
    parser.add_argument("--batch-size",    type=int, default=4)
    parser.add_argument("--grad-accum",    type=int, default=8)
    parser.add_argument("--steps",         type=int, default=1000)
    args = parser.parse_args()
    main(args)
