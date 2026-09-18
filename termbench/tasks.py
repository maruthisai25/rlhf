"""Task schema and jsonl IO.

A task is fully described by three bash snippets plus metadata:

  setup : run once in the fresh sandbox before the agent starts (creates fixtures)
  check : run after the agent finishes; exit code 0 == task solved
  instruction : natural-language request shown to the agent

Tasks are generated programmatically (scripts/gen_tasks.py) so ground truth is exact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Task:
    id: str
    category: str
    instruction: str
    setup: str
    check: str
    difficulty: int = 1
    max_turns: int = 8
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(**{k: d[k] for k in Task.__dataclass_fields__ if k in d})


def load_tasks(path: str | Path) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks


def save_tasks(tasks: list[Task], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
