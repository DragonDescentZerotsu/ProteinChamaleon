"""
Stage II — Mixed alignment + interleaved training.

Trains ProteinChameleonForCausalLM on two dataset types jointly:

  Alignment:    [BOS] Organism: ...\nSequence: ...\n<PROT_START> struct_tokens <PROT_END> function_text [EOS]
                Loss only over function_text tokens.

  Interleaved:  [BOS] Organism: ...\nSequence: ...\n\nnarrative with inline <PROT_START> ... <PROT_END> [EOS]
                Loss over the full narrative (text + structure tokens).

Samples are mixed proportional to dataset token count. All parameters are
unfrozen (full fine-tune from Stage I warmup checkpoint).

Sequence packing:
  Examples are greedily packed into fixed-length windows (default 4096) to
  avoid wasted padding. Loss mask prevents cross-example attention from leaking.

Usage:
    python scripts/train_stage2.py \
        --warmup-ckpt checkpoints/warmup/final \
        --align-dir  /data/steven/ProteinChamaleon/encoded/stage2/stage2-alignment \
        --interleaved-dir /data/steven/ProteinChamaleon/encoded/stage2 \
        --out-dir    checkpoints/stage2
"""

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import TrainingArguments, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ProteinChameleonTokenizer, ProteinChameleonForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage2")


# ── Individual example encoding ───────────────────────────────────────────────

def encode_alignment(
    organism: str,
    sequence: str,
    struct_bpe_ids: list[int],
    function_text: str,
    tokenizer: ProteinChameleonTokenizer,
    max_length: int,
) -> Optional[dict]:
    """
    Build token_ids and loss_mask for one alignment example.
    Loss is 1 only over function_text tokens (including EOS).
    """
    offset = tokenizer.protein_token_offset

    prefix_text  = f"Organism: {organism}\nSequence: {sequence}\n"
    prefix_ids   = ([tokenizer.text_tokenizer.bos_token_id]
                    + tokenizer.encode_text(prefix_text))
    struct_ids   = ([tokenizer.prot_start_id]
                    + [offset + i for i in struct_bpe_ids]
                    + [tokenizer.prot_end_id])
    suffix_ids   = (tokenizer.encode_text(function_text)
                    + [tokenizer.eos_id])

    input_ids  = prefix_ids + struct_ids + suffix_ids
    loss_mask  = ([0] * len(prefix_ids)
                  + [0] * len(struct_ids)
                  + [1] * len(suffix_ids))

    if len(input_ids) > max_length:
        return None

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }


_PROT_PATTERN = re.compile(r'\[PROT_START\](.*?)\[PROT_END\]', re.DOTALL)


def encode_interleaved(
    narrative: str,
    tokenizer: ProteinChameleonTokenizer,
    max_length: int,
) -> Optional[dict]:
    """
    Build token_ids and loss_mask for one interleaved example.
    narrative already contains the full text with inline [PROT_START] int... [PROT_END].
    Loss is 1 everywhere (text + structure tokens are all targets).
    """
    offset = tokenizer.protein_token_offset
    bos    = tokenizer.text_tokenizer.bos_token_id

    input_ids: list[int] = [bos]
    last = 0

    for m in _PROT_PATTERN.finditer(narrative):
        # text segment before this structure block
        text_seg = narrative[last:m.start()]
        if text_seg:
            input_ids.extend(tokenizer.encode_text(text_seg))

        # structure block
        raw_ids = [int(x) for x in m.group(1).split() if x.strip().isdigit()]
        input_ids.append(tokenizer.prot_start_id)
        input_ids.extend(offset + i for i in raw_ids)
        input_ids.append(tokenizer.prot_end_id)

        last = m.end()

    # trailing text
    tail = narrative[last:]
    if tail:
        input_ids.extend(tokenizer.encode_text(tail))
    input_ids.append(tokenizer.eos_id)

    if len(input_ids) > max_length:
        return None

    return {
        "input_ids": input_ids,
        "loss_mask": [1] * len(input_ids),
    }


# ── Datasets ──────────────────────────────────────────────────────────────────

class AlignmentDataset(Dataset):
    def __init__(self, npz_path: Path, tokenizer: ProteinChameleonTokenizer, max_length: int):
        d = np.load(npz_path, allow_pickle=True)
        self.token_ids     = d["token_ids"]
        self.accessions    = d["accessions"]
        self.sequences     = d["sequences"]
        self.function_text = d["function_text"]
        self.organism      = d["organism"]
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        logger.info("Alignment dataset: %d proteins from %s", len(self.token_ids), npz_path.name)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return encode_alignment(
            organism      = str(self.organism[idx]),
            sequence      = str(self.sequences[idx]),
            struct_bpe_ids= self.token_ids[idx].tolist(),
            function_text = str(self.function_text[idx]),
            tokenizer     = self.tokenizer,
            max_length    = self.max_length,
        )


class InterleavedDataset(Dataset):
    def __init__(self, npz_path: Path, tokenizer: ProteinChameleonTokenizer, max_length: int):
        d = np.load(npz_path, allow_pickle=True)
        self.narratives = d["narratives"]
        self.tokenizer  = tokenizer
        self.max_length = max_length
        logger.info("Interleaved dataset: %d proteins from %s", len(self.narratives), npz_path.name)

    def __len__(self):
        return len(self.narratives)

    def __getitem__(self, idx):
        return encode_interleaved(
            narrative  = str(self.narratives[idx]),
            tokenizer  = self.tokenizer,
            max_length = self.max_length,
        )


class ConcatDataset(Dataset):
    """Concatenate alignment + interleaved with per-example weights for sampling."""

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.lengths  = [len(d) for d in datasets]
        self.offsets  = [0]
        for l in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + l)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, (off, length) in enumerate(zip(self.offsets, self.lengths)):
            if idx < off + length:
                return self.datasets[i][idx - off]
        raise IndexError(idx)


# ── Sequence packing collator ─────────────────────────────────────────────────

class PackingCollator:
    """
    Greedily pack examples into fixed-length windows.
    Cross-example positions are zeroed in the loss mask.
    None examples (too long) are silently dropped.
    """

    def __init__(self, pad_id: int, max_length: int):
        self.pad_id     = pad_id
        self.max_length = max_length

    def __call__(self, features: list) -> dict:
        packed_ids   = []
        packed_mask  = []  # loss mask
        packed_attn  = []  # attention mask (1 for real tokens, 0 for pad)

        current_ids  = []
        current_loss = []

        for f in features:
            if f is None:
                continue
            ids  = f["input_ids"]
            lm   = f["loss_mask"]

            if len(current_ids) + len(ids) > self.max_length:
                # flush current window
                pad_len = self.max_length - len(current_ids)
                packed_ids.append(current_ids  + [self.pad_id] * pad_len)
                packed_mask.append(current_loss + [0]          * pad_len)
                packed_attn.append([1] * len(current_ids) + [0] * pad_len)
                current_ids  = []
                current_loss = []

            current_ids.extend(ids)
            current_loss.extend(lm)

        # flush last window
        if current_ids:
            pad_len = self.max_length - len(current_ids)
            packed_ids.append(current_ids  + [self.pad_id] * pad_len)
            packed_mask.append(current_loss + [0]          * pad_len)
            packed_attn.append([1] * len(current_ids) + [0] * pad_len)

        if not packed_ids:
            # all examples were None (too long) — return a dummy batch
            dummy = [self.pad_id] * self.max_length
            packed_ids  = [dummy]
            packed_mask = [[0] * self.max_length]
            packed_attn = [[0] * self.max_length]

        return {
            "input_ids":      torch.tensor(packed_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(packed_attn, dtype=torch.long),
            "loss_mask":      torch.tensor(packed_mask, dtype=torch.long),
        }


# ── Custom Trainer ────────────────────────────────────────────────────────────

class Stage2Trainer(Trainer):
    """Standard CE loss gated by per-token loss_mask from the collator."""

    def __init__(self, *args, weighted_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._weighted_sampler = weighted_sampler

    def get_train_dataloader(self) -> DataLoader:
        if self._weighted_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._weighted_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        loss_mask = inputs.pop("loss_mask")          # [B, T]
        outputs   = model(**inputs)
        logits    = outputs.logits                   # [B, T, V]

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].float().contiguous()   # [B, T-1, V]
        shift_labels = inputs["input_ids"][:, 1:].contiguous()  # [B, T-1]
        shift_mask   = loss_mask[:, 1:].contiguous()            # [B, T-1]

        # mask non-supervised positions
        labels = shift_labels.clone()
        labels[shift_mask == 0] = -100

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


# ── Weighted sampler ──────────────────────────────────────────────────────────

def make_weighted_sampler(datasets: list[Dataset], dataset_token_counts: list[int]) -> WeightedRandomSampler:
    """
    Sample proportional to token count so each dataset contributes
    ~proportionally to training compute.
    """
    total_tokens = sum(dataset_token_counts)
    weights = []
    for ds, tok_count in zip(datasets, dataset_token_counts):
        w = tok_count / total_tokens / len(ds)
        weights.extend([w] * len(ds))
    return WeightedRandomSampler(weights, num_samples=sum(len(d) for d in datasets), replacement=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = ProteinChameleonTokenizer.from_pretrained(args.warmup_ckpt)

    # ── Datasets ──────────────────────────────────────────────────────────────
    align_dir       = Path(args.align_dir)
    interleaved_dir = Path(args.interleaved_dir)

    align_train = AlignmentDataset(align_dir / "alignment_train.npz",       tokenizer, args.max_length)
    align_val   = AlignmentDataset(align_dir / "alignment_val.npz",         tokenizer, args.max_length)
    inter_train = InterleavedDataset(interleaved_dir / "interleaved_train.npz", tokenizer, args.max_length)
    inter_val   = InterleavedDataset(interleaved_dir / "interleaved_val.npz",   tokenizer, args.max_length)

    train_dataset = ConcatDataset([align_train, inter_train])
    val_dataset   = ConcatDataset([align_val,   inter_val])

    # Token counts for proportional sampling (alignment ~3.6x interleaved)
    ALIGN_TOKENS      = 392_314_473
    INTERLEAVED_TOKENS = 110_460_783

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading Stage I checkpoint from %s", args.warmup_ckpt)
    model = ProteinChameleonForCausalLM.from_pretrained(
        args.warmup_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Full fine-tune — unfreeze everything
    for param in model.parameters():
        param.requires_grad_(True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.2fB", n_params / 1e9)

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = PackingCollator(pad_id=tokenizer.pad_id, max_length=args.max_length)

    # ── Weighted sampler for proportional mixing ───────────────────────────────
    sampler = make_weighted_sampler(
        [align_train, inter_train],
        [ALIGN_TOKENS, INTERLEAVED_TOKENS],
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = str(out_dir),
        max_steps                   = args.steps,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        eval_strategy               = "steps",
        eval_steps                  = 200,
        save_steps                  = 500,
        logging_steps               = 10,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = args.warmup_steps,
        weight_decay                = 0.1,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.95,
        max_grad_norm               = 1.0,
        bf16                        = True,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_num_workers      = 4,
        remove_unused_columns       = False,
        report_to                   = "wandb",
        run_name                    = "stage2-mixed",
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
    )

    trainer = Stage2Trainer(
        model             = model,
        args              = training_args,
        train_dataset     = train_dataset,
        eval_dataset      = val_dataset,
        data_collator     = collator,
        weighted_sampler  = sampler,
    )

    logger.info("Starting Stage II training for %d steps", args.steps)
    logger.info(
        "Dataset: %d alignment + %d interleaved = %d total train examples",
        len(align_train), len(inter_train), len(train_dataset),
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    logger.info("Done. Saved to %s", out_dir / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ckpt",    default="checkpoints/warmup/final")
    parser.add_argument("--align-dir",      default="/data/steven/ProteinChamaleon/encoded/stage2/stage2-alignment")
    parser.add_argument("--interleaved-dir",default="/data/steven/ProteinChamaleon/encoded/stage2")
    parser.add_argument("--out-dir",        default="checkpoints/stage2")
    parser.add_argument("--max-length",     type=int,   default=4096)
    parser.add_argument("--batch-size",     type=int,   default=2,
                        help="Per-device batch size (each example is a packed window)")
    parser.add_argument("--grad-accum",     type=int,   default=16,
                        help="Gradient accumulation steps → effective batch ~128k tokens")
    parser.add_argument("--steps",          type=int,   default=10_000)
    parser.add_argument("--lr",             type=float, default=5e-5)
    parser.add_argument("--warmup-steps",   type=int,   default=500)
    args = parser.parse_args()
    main(args)
