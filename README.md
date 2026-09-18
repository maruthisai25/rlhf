# termbench — SFT → GRPO/RLAIF for a 0.8B terminal agent

A fully local post-training study on one RTX 4090: take **Qwen3.5-0.8B**, make it solve
multi-turn shell tasks in a sandbox, then measure what **supervised fine-tuning** and
**GRPO with an AI judge** each contribute. Everything (task generator, sandbox, agent loop,
teacher distillation, SFT, GRPO, benchmark, figures) is in this repository and reproducible
with one script. Start with **[EXPLAINED.md](EXPLAINED.md)** (why/what/how of every step,
in order); the paper-style write-up with all numbers is **[REPORT.md](REPORT.md)**.

<p align="center">
  <img src="figures/fig1_pass_rate.png" width="88%" alt="Pass rate by training stage">
</p>

## Results

| Stage | Adapter | Single-step pass (n=80) | Calls | Composite pass (n=48) | Calls | Judge |
|---|---|---|---|---|---|---|
| Base Qwen3.5-0.8B | – | 38.8% [29, 50] | 6.05 | 33.3% [22, 47] | 6.50 | 0.38 |
| SFT | `runs/sft/final` | **97.5%** [91, 99] | 2.20 | **83.3%** [70, 91] | 3.77 | 0.97 |
| GRPO run 1 | `sft → runs/grpo/final` | 93.8% [86, 97] | 1.34 | 79.2% [66, 88] | 2.71 | 0.94 |
| GRPO run 2 (RLAIF) | `sft → runs/grpo2/final` | **98.8%** [93, 100] | 1.77 | **83.3%** [70, 91] | 2.73 | 0.99 |

Brackets are Wilson 95% confidence intervals; *Judge* is the mean Qwen3.5-9B rubric score
in [0, 1] on the single-step set. Per-category tables: `python scripts/compare.py results`.

**Takeaways**

- **SFT does the heavy lifting.** 320 short teacher trajectories move the model from 39% to
  97.5%. It learns the tool-call format, to *write* the answer instead of stating it, and to stop.
- **GRPO run 2 keeps accuracy and buys efficiency.** Same pass rate as SFT within noise,
  20-28% fewer tool calls, and 100% of episodes end with a proper final message.
- **GRPO run 1 is a worked example of reward-noise amplification.** With the training set
  saturated after SFT, group-std normalisation turned tiny shaping terms into full-size
  advantages; the policy drifted to long prose and lost 3-4 points. Run 2 fixed it with
  Dr. GRPO normalisation, a pass-gated reward, a small KL term and harder composite tasks.
- **Evaluation pitfall.** A GRPO LoRA trained on the merged SFT model must be evaluated with
  the SFT adapter merged first (`--adapter runs/sft/final,runs/grpo2/final`), or scores
  silently collapse to baseline.

<p align="center">
  <img src="figures/fig2_behaviour.png" width="100%" alt="Behavioural metrics">
</p>
<p align="center">
  <img src="figures/fig5_grpo_dynamics.png" width="100%" alt="GRPO training dynamics">
</p>
<table align="center"><tr>
  <td><img src="figures/fig4_sft_loss.png" alt="SFT loss"></td>
  <td><img src="figures/fig3_category_heatmap.png" alt="Per-category pass rate"></td>
</tr></table>

All figures are in [`figures/`](figures) as 300-dpi PNG and vector PDF, regenerated from
`results/` and `logs/` by `scripts/make_figures.py`.

## How it works

| Component | Choice |
|---|---|
| Policy | `Qwen/Qwen3.5-0.8B`, bf16, LoRA on all linear layers (r=16 SFT, r=32 GRPO) |
| Teacher and judge | `Qwen3.5-9B` UD-Q4_K_XL on `llama-server` (Windows, port 8080) |
| Environment | one `bash` tool, fresh sandbox dir per episode, ≤ 8 turns, timeout + ulimits + denylist |
| Tasks | 33 templates with exact bash checkers → 320 train / 80 test; 112 / 48 composite two-part tasks |
| Reward (run 2) | `pass · (1 + 0.3·judge + 0.05·final_msg) − 0.02·max(0, calls − 4)` |
| Training | TRL `SFTTrainer` (assistant-only loss); TRL `GRPOTrainer` + `environment_factory`, tool results masked |
| Stack | WSL Ubuntu, torch 2.14 cu130, transformers 5.17, TRL 1.13, PEFT 0.21 |

```
termbench/            library: sandbox, tasks, env (TermEnv), judge client, agent loop, benchmark
scripts/
  run_all.sh          the whole pipeline, stage by stage, re-runnable
  gen_tasks.py        task generator (+ --hard for composite tasks), --validate runs every checker
  filter_hard.py      keep composite tasks the teacher can solve; build the GRPO train mix
  make_sft_data.py    teacher trajectories → data/sft.jsonl
  train_sft.py        stage 1
  train_grpo.py       stage 2 (docstring maps GRPO onto PPO terms)
  run_benchmark.py    evaluate base / adapter chain / teacher
  make_figures.py     figures/ + results table
  compare.py, status.sh, kill_job.sh, smoke_test.py, serve_judge.ps1
tasks/  data/  runs/*/final  results/*.episodes.jsonl  logs/  figures/  REPORT.md
```

## Reproduce

```powershell
# Windows: start the judge/teacher (downloads: unsloth/Qwen3.5-9B-GGUF UD-Q4_K_XL → ~/models/judge)
.\scripts\serve_judge.ps1
```

```bash
# WSL Ubuntu
uv venv --python 3.12 ~/rlhf/.venv && source ~/rlhf/.venv/bin/activate
uv pip install -r requirements.txt
python scripts/smoke_test.py                       # loads the policy, one generation, parser check
nohup setsid bash scripts/run_all.sh > logs/run_all.log 2>&1 &   # ~2 h on a 4090
bash scripts/status.sh                             # progress + GPU
```

`run_all.sh` runs every stage in order, including GRPO run 1 as the control arm, and skips a
stage when its output already exists. All outputs are committed, so in a fresh clone nothing
runs until you ask for it: `FRESH=1 bash scripts/run_all.sh` wipes the derived artefacts
(tasks, SFT data, adapters, results, logs) and reproduces the whole thing. Every benchmark
writes `results/<name>.episodes.jsonl` (full transcripts) and `results/<name>.summary.json`;
`make_figures.py` refuses to draw with any stage missing and writes byte-stable PDFs.

## Notes

- **VRAM.** Over-committing the card (> ~22 GB) pages the Vulkan judge out and slows it
  ~200×. Run one GPU stage at a time next to the judge; `run_all.sh` already does.
- **Speed.** Per-token latency of the hybrid architecture dominates (no `causal_conv1d`
  kernel); benchmarks step 32 episodes in lockstep, which is why they take minutes not hours.
- **Safety.** The sandbox is a guard rail, not a security boundary: commands run as the WSL
  user in a temp dir with `HOME` pointed at it, a 10 s timeout, ulimits and a denylist
  (`rm`/`mv`/`find` on absolute paths, `sudo`, `mkfs`, network, package installs).
- **Limitations.** Small test sets (differences < ~8 points are not significant), template
  tasks, one seed, judge and policy from the same model family. See REPORT.md §6.
