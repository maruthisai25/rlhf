"""Stage 2: GRPO with a live sandbox environment and an AI judge (RLAIF).

How this maps onto the RL you already know
------------------------------------------
* Policy pi_theta = the SFT model. An "action" is a whole assistant turn (tool call or final
  message); an episode is the multi-turn trajectory until the model stops calling tools.
* Reward is terminal and sparse-ish: checker pass (verifiable, from the environment) plus a
  0..1 score from the 9B judge (the "AI feedback" in RLAIF), plus tiny shaping terms.
  See termbench.env.TermEnv.get_reward.
* GRPO = PPO without a critic. For each prompt we sample G completions, and the advantage of
  completion i is (r_i - mean(r)) / std(r) over the group. That group baseline replaces the
  value function, which is why credit assignment works without TD learning: the whole
  sequence gets the same scalar advantage, applied to every one of its tokens.
* The loss is the PPO clipped surrogate on the per-token importance ratio
  pi_theta(token)/pi_old(token), clipped to [1-eps, 1+eps] (eps = 0.2 here). TRL's "dapo"
  variant normalises by total tokens across the batch instead of per-sequence.
* Tool results are environment observations, not policy outputs, so they are masked from
  the loss; only the model's own tokens receive the policy-gradient signal.
* beta=0: no KL penalty against a reference model (saves memory; LoRA already limits drift).

    python scripts/train_grpo.py --adapter runs/sft/final --out runs/grpo          # = run 1
    (run 2's flags: scripts/run_all.sh, stage 3b)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from termbench.env import TermEnv, build_prompt
from termbench.sandbox import Sandbox
from termbench.tasks import load_tasks

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--adapter", default=None, help="SFT LoRA adapter to merge before RL")
ap.add_argument("--split", default="tasks/train.jsonl")
ap.add_argument("--out", default="runs/grpo")
# Defaults reproduce GRPO run 1 exactly (see scripts/run_all.sh for both runs' full commands).
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--num-gens", type=int, default=8)
ap.add_argument("--gen-batch", type=int, default=16, help="rollouts per generation round (multiple of num-gens)")
ap.add_argument("--micro-batch", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--judge-weight", type=float, default=0.3)
ap.add_argument("--max-completion", type=int, default=1024)
ap.add_argument("--max-iters", type=int, default=6, help="max tool calls per rollout")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=0.0)
ap.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"],
                help="advantage normalisation: 'group' divides by the group std (amplifies noise on "
                     "saturated groups), 'batch' uses the batch std, 'none' = Dr. GRPO style")
ap.add_argument("--reward-version", default="v1", choices=["v1", "v2"])
ap.add_argument("--loss-type", default="dapo", choices=["grpo", "dapo", "dr_grpo"],
                help="'dr_grpo' with --scale-rewards none removes the length/difficulty biases")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

# ---- dataset: prompt built from a throwaway sandbox so the listing matches what reset() makes
tasks = load_tasks(a.split)[: a.limit]
rows = []
for t in tasks:
    with Sandbox() as sb:
        sb.run(t.setup, timeout=30, check_policy=False)
        listing = sb.snapshot()
    rows.append({"prompt": build_prompt(t.instruction, listing), "setup": t.setup, "check": t.check,
                 "instruction": t.instruction, "task_id": t.id})
ds = Dataset.from_list(rows).shuffle(seed=0)
print(f"{len(ds)} training prompts")

# ---- policy
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
if a.adapter:
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    print("merged SFT adapter", a.adapter)

TermEnv.judge_weight = a.judge_weight
TermEnv.reward_version = a.reward_version
if a.judge_weight > 0:
    from termbench.judge import judge_url

    print("judge:", judge_url())

peft_config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
                         task_type="CAUSAL_LM", target_modules="all-linear")

cfg = GRPOConfig(
    output_dir=a.out,
    max_steps=a.steps,
    learning_rate=a.lr,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=5,
    per_device_train_batch_size=a.micro_batch,
    gradient_accumulation_steps=a.gen_batch // a.micro_batch,
    generation_batch_size=a.gen_batch,
    num_generations=a.num_gens,
    max_completion_length=a.max_completion,
    max_tool_calling_iterations=a.max_iters,
    chat_template_kwargs={"enable_thinking": False},
    temperature=a.temperature,
    top_p=1.0,
    top_k=50,
    loss_type=a.loss_type,
    beta=a.beta,
    epsilon=0.2,
    scale_rewards={"group": "group", "batch": "batch", "none": False}[a.scale_rewards],
    mask_truncated_completions=True,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    save_steps=20,
    save_total_limit=2,
    report_to="none",
    log_completions=True,
    num_completions_to_print=1,
    seed=0,
)

trainer = GRPOTrainer(
    model=model, args=cfg, train_dataset=ds, processing_class=tok,
    environment_factory=TermEnv, peft_config=peft_config,
)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(os.path.join(a.out, "final"))
tok.save_pretrained(os.path.join(a.out, "final"))
print("saved adapter ->", os.path.join(a.out, "final"))
