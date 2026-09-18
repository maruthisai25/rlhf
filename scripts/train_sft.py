"""Stage 1: supervised fine-tuning (LoRA) on teacher trajectories.

Loss is computed on assistant tokens only (the model's own reasoning-free tool calls and final
messages); the user prompt, tool definitions and tool results are masked.

    python scripts/train_sft.py --data data/sft.jsonl --out runs/sft
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sft.jsonl")
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--out", default="runs/sft")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--max-length", type=int, default=2048)
ap.add_argument("--eval-frac", type=float, default=0.05)
a = ap.parse_args()

ds = load_dataset("json", data_files=a.data, split="train")
ds = ds.remove_columns([c for c in ds.column_names if c not in ("messages", "tools")])
split = ds.train_test_split(test_size=a.eval_frac, seed=0) if a.eval_frac > 0 else {"train": ds, "test": None}
print(f"train={len(split['train'])} eval={len(split['test']) if split['test'] is not None else 0}")

tok = AutoTokenizer.from_pretrained(a.model)
# Load the text-only causal LM (skips the vision tower of Qwen3.5) and pass the object to TRL.
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)

peft_config = LoraConfig(
    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules="all-linear",
)

cfg = SFTConfig(
    output_dir=a.out,
    num_train_epochs=a.epochs,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    per_device_train_batch_size=a.batch,
    gradient_accumulation_steps=a.grad_accum,
    per_device_eval_batch_size=a.batch,
    max_length=a.max_length,
    assistant_only_loss=True,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    eval_strategy="epoch" if split["test"] is not None else "no",
    save_strategy="no",
    report_to="none",
    seed=0,
)

trainer = SFTTrainer(
    model=model, args=cfg, train_dataset=split["train"], eval_dataset=split["test"],
    processing_class=tok, peft_config=peft_config,
)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(os.path.join(a.out, "final"))
tok.save_pretrained(os.path.join(a.out, "final"))
print("saved adapter ->", os.path.join(a.out, "final"))
