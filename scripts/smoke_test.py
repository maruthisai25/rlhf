"""Smoke test: load the policy, render a tool prompt, generate (cold + warm), parse the tool
call, and verify TRL's response-schema parser understands Qwen3.5's tool-call format.

    ~/rlhf/.venv/bin/python scripts/smoke_test.py [model_path]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from termbench.agent import HFBackend
from termbench.env import build_prompt, tool_schema

model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"

t0 = time.time()
be = HFBackend(model_path, max_new_tokens=96)
print(f"loaded {type(be.model).__name__} in {time.time()-t0:.1f}s; mem GB: {torch.cuda.memory_allocated()/1e9:.2f}")
print("generation_config eos:", be.model.generation_config.eos_token_id, "| using:", be.gen_kwargs["eos_token_id"])

messages = build_prompt("Create a file named hello.txt containing the word hello.", ".\n./notes.md")
for i in range(3):
    t0 = time.time()
    msg = be.step(messages, tool_schema())
    dt = time.time() - t0
    n = len(be.tok(msg["_raw"])["input_ids"])
    print(f"gen {i}: {n} tokens in {dt:.2f}s ({n/dt:.1f} tok/s) -> {msg.get('tool_calls')} | content={msg['content']!r}")

# TRL's parser (used inside GRPOTrainer's tool loop)
try:
    from trl.chat_template_utils import add_response_schema

    tok2 = add_response_schema(AutoTokenizer.from_pretrained(model_path))
    sample = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
    print("TRL parse_response:", tok2.parse_response(sample, prefix=""))
    print("TRL parse_response (model output):", tok2.parse_response(msg["_raw"], prefix=""))
except Exception as e:  # noqa: BLE001
    print("TRL response schema check failed:", type(e).__name__, e)
