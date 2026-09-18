"""termbench: a tiny multi-turn terminal-agent benchmark + RL environment.

Pipeline: base model -> benchmark -> SFT -> benchmark -> GRPO (RLAIF) -> benchmark.

Modules
-------
sandbox   : run bash commands in an isolated per-episode working directory (Linux / WSL)
tasks     : Task dataclass and jsonl loading
env       : TermEnv, the per-rollout environment used by TRL's GRPOTrainer *and* by the
            standalone agent loop, so training and evaluation see the same tool semantics
judge     : OpenAI-compatible client for the local llama-server judge (Qwen3.5-9B)
agent     : model-agnostic multi-turn agent loop (HF transformers backend or llama-server)
benchmark : run the agent over a task split and write results
"""
