"""Model-agnostic multi-turn agent loop.

Backends:
  HFBackend     : transformers model (optionally with a PEFT adapter) generating locally.
  ServerBackend : any OpenAI-compatible server (llama-server), used for the 9B teacher.

Both produce the same transcript format (OpenAI-style messages with `tool_calls` and `tool`
role messages), which is what TRL's SFTTrainer expects for tool-calling data and what
GRPOTrainer's tool loop produces internally. Keeping the format identical across teacher,
SFT, RL and eval is the whole point.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .env import TermEnv, build_prompt, tool_schema
from .tasks import Task

# Qwen3.5 chat template renders tool calls as:
#   <tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>
# Older Qwen templates used <tool_call>{"name": ..., "arguments": {...}}</tool_call>. Accept both.
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
FUNC_RE = re.compile(r"<function=([\w\-.]+)>(.*?)(?:</function>|$)", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([\w\-.]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _parse_one(raw: str) -> dict | None:
    raw = raw.strip()
    fm = FUNC_RE.search(raw)
    if fm:
        name = fm.group(1)
        args = {k: v for k, v in PARAM_RE.findall(fm.group(2))}
        if not args:  # unterminated parameter block: take everything after the tag
            pm = re.search(r"<parameter=([\w\-.]+)>\n?(.*)", fm.group(2), re.DOTALL)
            if pm:
                args = {pm.group(1): pm.group(2).strip()}
        return {"name": name, "arguments": args}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m2 = re.search(r'"command"\s*:\s*"(.*)"\s*\}\s*\}?\s*$', raw, re.DOTALL)
            if not m2:
                return None
            obj = {"name": "bash", "arguments": {"command": m2.group(1)}}
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"command": args}
        return {"name": obj.get("name"), "arguments": args}
    return None


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split raw generated text into (content, tool_calls)."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        obj = _parse_one(m.group(1))
        if obj and obj.get("name"):
            calls.append({"type": "function", "function": {"name": obj["name"], "arguments": obj["arguments"]}})
    content = TOOL_CALL_RE.sub("", text)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content, calls


@dataclass
class Episode:
    task_id: str
    category: str
    messages: list[dict]
    calls: list[dict]
    result: dict
    turns: int
    seconds: float
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


class HFBackend:
    def __init__(self, model_path: str, adapter: str | None = None, max_new_tokens: int = 384,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 20,
                 repetition_penalty: float = 1.05, device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype or torch.bfloat16, device_map=device
        )
        # `adapter` may be a comma-separated chain, e.g. "runs/sft/final,runs/grpo2/final": the GRPO
        # LoRA was trained on top of the *merged* SFT model, so the SFT adapter must be merged first.
        for path in [p for p in (adapter or "").split(",") if p.strip()]:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, path.strip()).merge_and_unload()
        self.model.eval()
        # Stop on either end-of-turn token; Qwen's generation_config normally lists both but
        # be explicit. Also stop as soon as a tool call closes: one tool call per turn.
        eos = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                eos.add(tid)
        self.gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            eos_token_id=sorted(eos),
            stop_strings=["</tool_call>"],
            tokenizer=self.tok,
        )
        self.name = model_path + (f"+{adapter}" if adapter else "")

    def step(self, messages: list[dict], tools: list[dict]) -> dict:
        return self.step_batch([messages], tools)[0]

    def step_batch(self, conversations: list[list[dict]], tools: list[dict]) -> list[dict]:
        """One assistant turn for several independent conversations, generated as one batch."""
        import torch

        texts = []
        for messages in conversations:
            clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
            texts.append(self.tok.apply_chat_template(
                clean, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False
            ))
        self.tok.padding_side = "left"
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, **self.gen_kwargs)
        start = enc["input_ids"].shape[1]
        results = []
        for r in range(len(texts)):
            raw = self.tok.decode(out[r, start:], skip_special_tokens=True)
            content, calls = parse_tool_calls(raw)
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls[:1]
            msg["_raw"] = raw
            results.append(msg)
        return results


class ServerBackend:
    """OpenAI-compatible chat backend (llama-server). Arguments come back as JSON strings."""

    def __init__(self, temperature: float = 0.3, max_tokens: int = 384, name: str = "server"):
        from . import judge

        self._chat = judge.chat
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = name

    def step(self, messages: list[dict], tools: list[dict]) -> dict:
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        # servers expect arguments as JSON strings in history
        for m in clean:
            if m.get("tool_calls"):
                m["tool_calls"] = [
                    {**tc, "id": tc.get("id", f"call_{i}"), "function": {**tc["function"],
                     "arguments": json.dumps(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], dict) else tc["function"]["arguments"]}}
                    for i, tc in enumerate(m["tool_calls"])
                ]
        msg = self._chat(clean, tools=tools, temperature=self.temperature, max_tokens=self.max_tokens)
        out: dict[str, Any] = {"role": "assistant", "content": (msg.get("content") or "").strip()}
        content2, inline_calls = parse_tool_calls(out["content"])
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"command": args}
            calls.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args}})
        if inline_calls and not calls:
            calls, out["content"] = inline_calls, content2
        if calls:
            out["tool_calls"] = calls
        return out


def run_episode(task: Task, backend, max_turns: int | None = None, env: TermEnv | None = None,
                verbose: bool = False) -> Episode:
    """Drive one task to completion with `backend`. Returns the transcript + evaluation."""
    t0 = time.time()
    env = env or TermEnv()
    env.reset(setup=task.setup, check=task.check, instruction=task.instruction, task_id=task.id)
    messages = build_prompt(task.instruction, env._listing())
    tools = tool_schema()
    limit = max_turns or task.max_turns
    error = None
    turns = 0
    try:
        for turns in range(1, limit + 1):
            msg = backend.step(messages, tools)
            messages.append(msg)
            if verbose:
                print(f"  [{turns}] {msg.get('_raw', msg.get('content'))!r}"[:300])
            calls = msg.get("tool_calls") or []
            if not calls:
                env.finished = True
                break
            for tc in calls:
                fn = tc["function"]
                if fn.get("name") != "bash":
                    obs = f"[exit 1] unknown tool {fn.get('name')!r}; only `bash` is available"
                else:
                    obs = env.bash(fn.get("arguments", {}).get("command", ""))
                messages.append({"role": "tool", "name": "bash", "content": obs})
                if verbose:
                    print(f"      $ {fn.get('arguments', {}).get('command', '')!r}\n      -> {obs[:200]!r}")
    except Exception as e:  # noqa: BLE001 - we want the benchmark to keep going
        error = f"{type(e).__name__}: {e}"
    result = env._evaluate(messages)
    env._close()
    return Episode(
        task_id=task.id, category=task.category, messages=messages, calls=list(env.calls),
        result=result, turns=turns, seconds=round(time.time() - t0, 2), error=error,
    )


def run_episodes_batched(tasks: list[Task], backend, batch_size: int = 8, max_turns: int | None = None,
                         on_done=None) -> list[Episode]:
    """Run many episodes in lockstep: each round generates one assistant turn for every active
    episode as a single padded batch, then executes the tool calls. Same semantics as
    `run_episode`, several times faster for the local HF policy."""
    from concurrent.futures import ThreadPoolExecutor

    tools = tool_schema()
    episodes: list[Episode | None] = [None] * len(tasks)
    queue = list(range(len(tasks)))
    active: list[dict] = []
    # Checker + judge scoring run off the generation thread so a wave of finishing episodes
    # (and a slow judge) never stalls the GPU.
    scorer = ThreadPoolExecutor(max_workers=4)
    pending = []

    def start(i: int) -> dict:
        task = tasks[i]
        env = TermEnv()
        env.reset(setup=task.setup, check=task.check, instruction=task.instruction, task_id=task.id)
        return {"i": i, "task": task, "env": env, "messages": build_prompt(task.instruction, env._listing()),
                "turns": 0, "t0": time.time(), "error": None, "limit": max_turns or task.max_turns}

    def _score(a: dict):
        task, env = a["task"], a["env"]
        result = env._evaluate(a["messages"])
        env._close()
        ep = Episode(task_id=task.id, category=task.category, messages=a["messages"], calls=list(env.calls),
                     result=result, turns=a["turns"], seconds=round(time.time() - a["t0"], 2), error=a["error"])
        episodes[a["i"]] = ep
        if on_done:
            on_done(ep)

    def finish(a: dict):
        pending.append(scorer.submit(_score, a))

    while queue or active:
        while queue and len(active) < batch_size:
            try:
                active.append(start(queue.pop(0)))
            except Exception as e:  # noqa: BLE001
                i = len(tasks) - len(queue) - 1
                episodes[i] = Episode(task_id=tasks[i].id, category=tasks[i].category, messages=[], calls=[],
                                      result={"passed": False, "n_calls": 0, "finished_cleanly": False, "judge": None, "reward": 0.0},
                                      turns=0, seconds=0.0, error=f"setup: {e}")
        if not active:
            break
        try:
            msgs = backend.step_batch([a["messages"] for a in active], tools)
        except Exception as e:  # noqa: BLE001
            for a in active:
                a["error"] = f"{type(e).__name__}: {e}"
                finish(a)
            active = []
            continue
        done = []
        for a, msg in zip(active, msgs):
            a["turns"] += 1
            a["messages"].append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                a["env"].finished = True
                done.append(a)
                continue
            for tc in calls:
                fn = tc["function"]
                if fn.get("name") != "bash":
                    obs = f"[exit 1] unknown tool {fn.get('name')!r}; only `bash` is available"
                else:
                    obs = a["env"].bash(fn.get("arguments", {}).get("command", ""))
                a["messages"].append({"role": "tool", "name": "bash", "content": obs})
            if a["turns"] >= a["limit"]:
                done.append(a)
        for a in done:
            finish(a)
            active.remove(a)
    for f in pending:
        f.result()
    scorer.shutdown()
    return [e for e in episodes if e is not None]
