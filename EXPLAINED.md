# termbench, explained

Every step of the project in order: what was done, why, and how. Read this first; `REPORT.md`
has the numbers, `README.md` has the commands.

## 0. Goal

Take a small open model, make it act as a terminal agent, measure it, improve it with
supervised fine-tuning (SFT), improve it again with reinforcement learning from AI feedback
(RLAIF), and measure after every stage. One RTX 4090, one day, nothing leaves the machine.

## 1. Model choice — Qwen3.5-0.8B

**Why small.** RL needs thousands of rollouts; at 0.8B a rollout costs seconds and the whole
run fits next to a 9B judge on 24 GB.
**Why this one.** Newest Qwen small line (Qwen3.6 has no small variants), instruction-tuned,
already speaks a tool-call format. It is a hybrid architecture (Gated DeltaNet + attention);
the only practical consequence was slow per-token generation without the `causal_conv1d`
CUDA kernel, which shaped how evaluation was batched (§7).

## 2. Teacher and judge — Qwen3.5-9B on llama.cpp

**Why one model for both.** It is free, local, and strong enough at shell tasks (solved
320/320 training tasks). Served once on `llama-server`, reached over HTTP from the training
process. Two roles: *teacher* generates SFT trajectories; *judge* scores transcripts 0-10 for
correctness, safety, efficiency, communication. The rubric answer is a JSON score parsed into
[0, 1].
**Caveat.** Judge and policy share a family, which favours format agreement. The checker,
not the judge, is ground truth everywhere in the results.

## 3. Tasks — generated, not collected

**Why generate.** A benchmark needs an exact pass/fail signal; hand-written shell datasets
have no checkers. Each of 33 templates emits a `(setup, instruction, check)` triple in bash:
`setup` builds fixtures with content chosen in Python, `check` is a shell test whose expected
value was computed in Python at generation time. `--validate` runs every setup and asserts
the check *fails* on the untouched fixture, so no task is trivially passed.
**Splits.** 320 train / 80 test with disjoint seeds, every template in both.
**Composite tasks.** After SFT the single-step tasks were saturated (§8), so a second
generator glues two templates into one instruction ("Do both of the following"), rejects
pairs that both write `answer.txt` or share fixture names, and keeps only those the teacher can
solve at least once in two tries (112 train / 48 test). These are the "hard" set.

## 4. Environment — one sandbox per episode, one class for everything

**What.** `TermEnv` exposes a single tool, `bash(command)`. Each episode gets a fresh temp
directory; commands run via `bash -c` with `HOME` pointed inside it, a 10 s timeout, ulimits,
and a denylist (absolute-path `rm`/`mv`/`find`, `sudo`, `mkfs`, network, installs). Output is
returned as `[exit N]` plus truncated stdout/stderr. The episode ends when the model replies
without a tool call, or at 8 turns.
**Why one class.** TRL's `GRPOTrainer` accepts an `environment_factory`: it calls
`reset(**row)`, exposes every *public* method as a tool through the chat template, and calls
`get_reward()` after the rollout. The same object drives the evaluation loop and SFT data
generation, so the tool semantics the model is trained on are exactly the ones it is scored
on. (Helper methods had to be underscore-prefixed, or TRL offered `close` and `evaluate` to
the model as tools.)

## 5. Benchmark protocol

Pass rate = checker exit 0. Also recorded: tool calls per episode, whether the episode ended
with a final message, and the judge score. Sampling at temperature 0.7, top-p 0.9, one tool
call per turn, `</tool_call>` as a stop string. Confidence intervals are Wilson 95%; with
80 and 48 tasks, differences under ~8 points are not significant and are reported as such.

## 6. Stage 1 — SFT by rejection-sampled distillation

**Why first.** The base model loops (`cat answer.txt` until the turn cap), refuses, and
*states* answers instead of writing them. Those are format and habit problems; imitation
fixes them cheaply.
**Data.** The teacher runs the same agent loop on the 320 training tasks; the first
trajectory that passes, ends cleanly and uses ≤ 6 calls is kept. Result: 320 trajectories,
mean 2.24 calls, short final messages. Stored in TRL's tool-calling conversation format
(assistant messages carry `tool_calls`, tool results are `tool` role).
**Training.** LoRA r=16 on all linear layers, loss on assistant tokens only (prompt, tool
schema and tool outputs masked), 3 epochs, lr 1e-4, cosine, effective batch 16. 6.7 minutes.
Eval loss 0.085 → 0.066.
**Effect.** 38.8% → 97.5% on single-step tasks, 6.05 → 2.20 calls, 100% clean finishes.
83.3% on composite tasks it never saw.

## 7. Making evaluation fast

Per-token latency of the hybrid model (14-47 tok/s) made sequential evaluation take hours,
and it was worse under GPU contention. Fix: run up to 32 episodes in lockstep, one padded
batched `generate` per turn, then execute all tool calls, then repeat. Checker and judge
calls moved to a thread pool so a wave of finishing episodes never stalls generation.
80 tasks now take about two minutes.

## 8. Stage 2 — GRPO with execution reward + AI judge

**Why GRPO.** It is PPO without a critic. For each prompt, sample G = 8 rollouts; the
advantage of rollout *i* is its reward relative to the group. The group mean replaces the
value function, so no separate model to train and no TD learning. The loss is the clipped
PPO surrogate on per-token importance ratios (ε = 0.2). Tool-result tokens are environment
observations, not policy output, so TRL masks them from the loss.
**Reward.** Checker pass is the verifiable term; the judge adds a graded RLAIF term so two
passing rollouts are not tied; small shaping terms for finishing cleanly and for efficiency.

### Run 1 — what went wrong

Config: 320 single-step prompts, `scale_rewards=group` (divide by group std), reward
`pass + 0.1·finished − 0.02·(calls−1) + 0.3·judge`, lr 2e-5, no KL.
Result: 93.8% / 79.2%, below SFT.
**Mechanism.** SFT already solved nearly every training prompt, so most groups had eight
passing rollouts. Dividing by a tiny group std turned the only remaining variation, the
per-call penalty and judge noise, into unit-scale advantages. The policy was pushed hard on
"fewer calls, whatever the judge liked" rather than on solving tasks. Completion length
drifted from ~100 to ~450 tokens; transcripts show prose before tool calls, stray `</think>`
tokens, hallucinated user turns, and answers described but not written. At the last step all
16 rollouts had identical reward and zero advantage: no signal left at all.

### Run 2 — the fix

Three changes, applied together:
1. **Harder prompts** (112 composite + 64 single-step) so groups have mixed outcomes.
2. **Dr. GRPO normalisation** (`loss_type=dr_grpo`, `scale_rewards=False`): no std division,
   no length bias. Plus β = 0.02 KL to the SFT policy and lr 1e-5 to limit drift.
3. **Pass-gated reward** `pass·(1 + 0.3·judge + 0.05·finished) − 0.02·max(0, calls−4)`:
   judge and shaping can only separate *passing* rollouts.
Result: 98.8% / 83.3% (SFT-level accuracy, within CI), 20-28% fewer tool calls, 100% clean
finishes, completion length stable at 120-180 tokens, KL ≈ 0.02 throughout. 18 minutes.

## 9. The evaluation bug worth remembering

The GRPO LoRA is trained on top of the *merged* SFT model. Evaluating base + GRPO adapter
alone scores 39-46%, indistinguishable from the base model, and that is exactly how the first
GRPO numbers came out. The evaluator now takes an adapter chain
(`--adapter runs/sft/final,runs/grpo2/final`) and merges in order. Every GRPO number in the
report is from the corrected re-run.

## 10. What each stage bought

| stage | pass (easy / hard) | calls (easy / hard) | cost |
|---|---|---|---|
| base | 38.8% / – | 6.05 / – | – |
| SFT | 97.5% / 83.3% | 2.20 / 3.77 | 25 min data + 7 min train |
| GRPO run 2 | 98.8% / 83.3% | 1.77 / 2.73 | 10 min hard tasks + 18 min train |

SFT bought accuracy. RL bought behaviour: fewer calls, always a final message, no
regressions. To make RL move accuracy, the next step is tasks the SFT model fails more often:
longer compositions, tasks that require inspecting before acting, held-out template families.
