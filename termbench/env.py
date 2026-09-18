"""TermEnv: the environment the agent acts in.

The same class serves two roles:

1. TRL GRPOTrainer `environment_factory`. TRL calls `reset(**dataset_row)` before each
   rollout, exposes every public method (here: `bash`) to the model as a tool via the chat
   template, and calls `get_reward()` once the rollout ends. The trainer masks the tool
   results from the loss automatically, so only the model's own tokens are trained on.

2. The standalone agent loop in `termbench.agent` (used for benchmarking and for producing
   SFT trajectories with the teacher model) drives exactly the same object, so the tool
   semantics the model is trained on are the ones it is evaluated on.

Reward (see `get_reward`):
    R = 1.0 * task_passed
      + 0.1 * finished_cleanly     (ended with a final message, used at least one tool call)
      - 0.02 * max(0, n_calls - 1) (mild efficiency pressure)
      + JUDGE_WEIGHT * judge_score (optional, RLAIF: llama-server Qwen3.5-9B rates the transcript)
"""
from __future__ import annotations

import json
import os
from typing import Any

from .sandbox import Sandbox

SYSTEM_PROMPT = (
    "You are a command-line agent working inside a Linux sandbox. "
    "You can run shell commands with the `bash` tool. Each call runs `bash -c` in the task's "
    "working directory (the directory never changes between calls, so `cd` does not persist). "
    "Inspect the environment when needed, complete the task, then reply with a one-line summary "
    "WITHOUT calling any tool. Do not ask the user questions. Do not explain unless asked."
)


def build_prompt(instruction: str, listing: str) -> list[dict]:
    """Messages shown to the agent at the start of an episode."""
    user = f"Task: {instruction}\n\nInitial listing of the working directory (depth 2):\n{listing}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class TermEnv:
    """One instance per rollout. Stateless between `reset` calls except for the sandbox."""

    judge_weight: float = float(os.environ.get("TERMBENCH_JUDGE_WEIGHT", "0.0"))
    # score with the judge even when it does not contribute to reward (benchmark reporting)
    judge_always: bool = os.environ.get("TERMBENCH_JUDGE", "0") == "1"
    # reward shaping version (see _evaluate). v1 = original; v2 = shaping only on passing rollouts.
    reward_version: str = os.environ.get("TERMBENCH_REWARD", "v1")

    def __init__(self):
        self.sandbox: Sandbox | None = None
        self.task: dict[str, Any] = {}
        self.calls: list[dict] = []
        self.finished = False
        self.last_reward: dict | None = None

    # ---- TRL reserved methods ----------------------------------------------------------
    def reset(self, setup: str = "", check: str = "", instruction: str = "", task_id: str = "", **kwargs) -> None:
        """Create a fresh sandbox and run the task's setup script.

        Returns None: the prompt is already in the dataset's `prompt` column (built with
        `build_prompt`, which needs the listing *before* TRL asks for a reset, so the listing is
        generated at dataset-build time from a throwaway sandbox; it is deterministic).
        """
        self._close()
        self.sandbox = Sandbox()
        self.task = {"setup": setup, "check": check, "instruction": instruction, "task_id": task_id, **kwargs}
        self.calls = []
        self.finished = False
        self.last_reward = None
        if setup:
            r = self.sandbox.run(setup, timeout=30, check_policy=False)
            if r.exit_code != 0:
                raise RuntimeError(f"task {task_id}: setup failed: {r.stderr[:500]}")
        return None

    def get_reward(self) -> float:
        """Score the finished rollout from environment state, then tear the sandbox down."""
        info = self._evaluate()
        self._close()
        return info["reward"]

    # ---- tools (every public method is exposed to the model) ----------------------------
    def bash(self, command: str) -> str:
        """Run a shell command in the task's working directory and return its output.

        Args:
            command: The bash command line to execute (may use pipes, redirects, &&, etc.).

        Returns:
            The exit code followed by stdout and stderr of the command (truncated if long).
        """
        if self.sandbox is None:
            return "[exit 1] environment not initialised"
        if not isinstance(command, str):
            command = json.dumps(command)
        result = self.sandbox.run(command)
        obs = result.observation()
        self.calls.append({"command": command, "exit_code": result.exit_code, "observation": obs})
        return obs

    # ---- helpers used by both TRL and the standalone loop ------------------------------
    # (underscore-prefixed: TRL exposes every *public* method of the environment as a tool)
    def _evaluate(self, transcript: list[dict] | None = None) -> dict:
        """Run the task's checker and compute the shaped reward. Does not close the sandbox."""
        assert self.sandbox is not None, "reset() first"
        chk = self.sandbox.run(self.task.get("check", "false"), timeout=30, check_policy=False)
        passed = chk.exit_code == 0
        n_calls = len(self.calls)
        finished_cleanly = self.finished and n_calls >= 1
        judge = None
        if self.judge_weight > 0 or self.judge_always:
            from .judge import score_transcript

            judge = score_transcript(self.task, self.calls, transcript, checker_passed=passed)
        j = judge if judge is not None else 0.0
        if self.reward_version == "v2":
            # Lesson from run 1: with group-normalised advantages, tiny shaping terms on saturated
            # groups (all rollouts pass) become full-size learning signal and the policy learned to
            # stop after one call. v2 makes every secondary term conditional on passing, and only
            # penalises genuinely long episodes.
            reward = passed * (1.0 + self.judge_weight * j + 0.05 * finished_cleanly) - 0.02 * max(0, n_calls - 4)
        else:
            reward = 1.0 * passed + 0.1 * finished_cleanly - 0.02 * max(0, n_calls - 1)
            if judge is not None and self.judge_weight > 0:
                reward += self.judge_weight * judge
        self.last_reward = {
            "passed": passed,
            "n_calls": n_calls,
            "finished_cleanly": finished_cleanly,
            "judge": judge,
            "reward": round(reward, 4),
            "check_stdout": chk.stdout[-300:],
            "check_stderr": chk.stderr[-300:],
        }
        return self.last_reward

    def _listing(self) -> str:
        assert self.sandbox is not None
        return self.sandbox.snapshot()

    def _close(self):
        if self.sandbox is not None:
            self.sandbox.close()
            self.sandbox = None

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass


def tool_schema() -> list[dict]:
    """OpenAI-style JSON schema for the `bash` tool (used by llama-server / teacher)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the task's working directory and return its output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command line to execute (may use pipes, redirects, &&, etc.).",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]
