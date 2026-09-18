"""Generate the report figures from results/*.episodes.jsonl, results/*.summary.json and logs/.

    python scripts/make_figures.py            # writes figures/*.png and *.pdf

Design: one baseline, thin marks, hairline recessive grid, fixed categorical hue order
(blue, orange, aqua), sequential blue for magnitude, text in ink tokens (never series color).
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LOGS = ROOT / "logs"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (validated reference instance) -------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3: blue, orange, aqua
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "text.color": INK,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
})

STAGES = OrderedDict([
    ("0_base", "Base\nQwen3.5-0.8B"),
    ("1_sft", "SFT"),
    ("2_grpo", "GRPO run 1"),
    ("3_grpo2", "GRPO run 2"),
])
HARD = {"0_base": "0_base_hard", "1_sft": "1_sft_hard", "2_grpo": "2_grpo_hard", "3_grpo2": "3_grpo2_hard"}
# Every stage above must have results; a missing one is an error, not a placeholder.
REQUIRED = list(STAGES) + list(HARD.values())


def strip(ax, y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not y:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_color(BASELINE)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. Run the corresponding stage of scripts/run_all.sh first.")
    return p


def episodes(name: str) -> list[dict]:
    p = _require(RES / f"{name}.episodes.jsonl")
    eps = [json.loads(l) for l in open(p, encoding="utf-8")]
    if not eps:
        raise ValueError(f"{p} has no episodes")
    return eps


def summary(name: str) -> dict:
    return json.load(open(_require(RES / f"{name}.summary.json")))


def check_inputs() -> None:
    missing = [n for n in REQUIRED if not (RES / f"{n}.episodes.jsonl").exists() or not (RES / f"{n}.summary.json").exists()]
    missing += [f"logs/{n}" for n in ("train_sft.log", "train_grpo.log", "train_grpo2.log") if not (LOGS / n).exists()]
    if missing:
        raise SystemExit("make_figures: missing inputs, refusing to draw partial figures:\n  " + "\n  ".join(missing))


def mean_ci(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), 0.0
    m = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / math.sqrt(len(xs)))
    return m, 1.96 * se


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    # No timestamps in the PDF so regenerating from unchanged results is byte-identical.
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---- Figure 1: pass rate by stage, easy vs hard, Wilson 95% CI ----------------------------
def fig_pass_rate():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = list(STAGES.values())
    x = np.arange(len(STAGES))
    w = 0.34
    from matplotlib.patches import Patch

    series_defs = [
        ("Single-step test (n = 80)", lambda k: k, SERIES[0]),
        ("Composite hard test (n = 48)", lambda k: HARD[k], SERIES[1]),
    ]
    for j, (series, key_fn, color) in enumerate(series_defs):
        for i, key in enumerate(STAGES):
            eps = episodes(key_fn(key))
            k = sum(e["result"]["passed"] for e in eps)
            p, lo, hi = wilson(k, len(eps))
            ax.bar(x[i] + (j - 0.5) * (w + 0.02), p, w, color=color, zorder=3)
            ax.errorbar(x[i] + (j - 0.5) * (w + 0.02), p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK2,
                        elinewidth=0.9, capsize=2.5, zorder=4)
            ax.text(x[i] + (j - 0.5) * (w + 0.02), hi + 0.02, f"{100*p:.1f}%", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Task pass rate (checker)")
    ax.set_title("Task success by training stage", loc="left")
    handles = [Patch(facecolor=c, label=s) for s, _, c in series_defs]
    fig.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.06))
    strip(ax)
    fig.text(0.01, -0.13, "Error bars: Wilson 95% confidence intervals. GRPO adapters are evaluated on top of the merged SFT model.",
             fontsize=7, color=MUTED)
    save(fig, "fig1_pass_rate")


# ---- Figure 2: secondary metrics (small multiples, single series) ------------------------
def fig_secondary():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    keys = list(STAGES)
    labels = ["Base", "SFT", "GRPO 1", "GRPO 2"]
    x = np.arange(len(keys))
    panels = [
        ("Tool calls per episode", lambda e: e["result"]["n_calls"], "mean", "{:.2f}"),
        ("Ends with a final message", lambda e: float(e["result"]["finished_cleanly"]), "rate", "{:.0%}"),
        ("Judge score (0-1)", lambda e: e["result"].get("judge"), "mean", "{:.2f}"),
    ]
    for ax, (title, fn, kind, fmt) in zip(axes, panels):
        for i, key in enumerate(keys):
            eps = episodes(key)
            vals = [fn(e) for e in eps]
            if kind == "rate":
                k = int(sum(vals))
                m, lo, hi = wilson(k, len(vals))
                err = [[m - lo], [hi - m]]
            else:
                m, half = mean_ci(vals)
                err = [[half], [half]]
            ax.bar(x[i], m, 0.55, color=SERIES[0], zorder=3)
            ax.errorbar(x[i], m, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=4)
            ax.text(x[i], m + (err[1][0] if err else 0) + (0.02 if kind == "rate" else 0.12 if "calls" in title else 0.02),
                    fmt.format(m), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left")
        strip(ax)
        if kind == "rate" or "Judge" in title:
            ax.set_ylim(0, 1.15)
            ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"] if "Judge" in title else ["0%", "50%", "100%"])
        else:
            ax.set_ylim(0, 8)
    fig.suptitle("Behavioural metrics on the single-step test set (n = 80 per stage)", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold")
    fig.text(0.01, -0.03, "Error bars: 95% CI (normal approximation for means, Wilson for rates). Judge = Qwen3.5-9B rubric score.",
             fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_behaviour")


# ---- Figure 3: per-category heatmap -------------------------------------------------------
def fig_heatmap():
    keys = list(STAGES)
    sums = [summary(k) for k in keys]
    cats = sorted({c for s in sums for c in s["by_category"]})
    M = np.full((len(cats), len(keys)), np.nan)
    N = np.zeros_like(M)
    for j, s in enumerate(sums):
        for i, c in enumerate(cats):
            if c in s["by_category"]:
                M[i, j] = s["by_category"][c]["pass_rate"]
                N[i, j] = s["by_category"][c]["n"]
    order = np.argsort(np.nan_to_num(M[:, 0]) + 0.01 * np.nan_to_num(M[:, 1]))
    cats = [cats[i] for i in order]
    M = M[order]
    fig, ax = plt.subplots(figsize=(5.2, 7.6))
    im = ax.imshow(M, cmap=BLUE_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys)), ["Base", "SFT", "GRPO 1", "GRPO 2"])
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(cats)):
        for j in range(len(keys)):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_ticks([0, 0.5, 1.0], labels=["0%", "50%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    ax.set_title("Pass rate by task category (single-step test set)", loc="left")
    fig.text(0.01, 0.005, "Cell value = percent of that category's test tasks passed (2-3 tasks per category).",
             fontsize=7, color=MUTED)
    save(fig, "fig3_category_heatmap")


# ---- Figure 4: SFT curves -----------------------------------------------------------------
def parse_dicts(path: Path) -> list[dict]:
    rows = []
    for line in open(_require(path), errors="replace"):
        for m in re.finditer(r"\{\x27(?:loss|eval_loss)\x27.*?\}", line):
            try:
                rows.append(ast.literal_eval(m.group(0)))
            except Exception:
                pass
    return rows


def fig_sft():
    rows = parse_dicts(LOGS / "train_sft.log")
    tr = [(float(r["epoch"]), float(r["loss"])) for r in rows if "loss" in r]
    ev = [(float(r["epoch"]), float(r["eval_loss"])) for r in rows if "eval_loss" in r]
    if not tr or not ev:
        raise ValueError("logs/train_sft.log has no loss / eval_loss records")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot([e for e, _ in tr], [l for _, l in tr], color=SERIES[0], lw=2, solid_capstyle="round", label="Training loss")
    ax.plot([e for e, _ in ev], [l for _, l in ev], color=SERIES[1], lw=0, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Eval loss (end of epoch)")
    for e, l in ev:
        ax.text(e + 0.04, l - 0.006, f"{l:.3f}", ha="left", va="top", fontsize=7.5, color=INK2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Token cross-entropy (assistant tokens only)")
    ax.set_title("Supervised fine-tuning on 320 teacher trajectories", loc="left")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    strip(ax)
    save(fig, "fig4_sft_loss")


# ---- Figure 5: GRPO dynamics run 1 vs run 2 ------------------------------------------------
def parse_grpo(path: Path) -> list[dict]:
    rows = []
    for line in open(_require(path), errors="replace"):
        for m in re.finditer(r"\{\x27loss\x27.*?\}", line):
            try:
                rows.append(ast.literal_eval(m.group(0)))
            except Exception:
                pass
    if not rows:
        raise ValueError(f"{path} has no per-step training records")
    return rows


def fig_grpo():
    r1 = parse_grpo(LOGS / "train_grpo.log")
    r2 = parse_grpo(LOGS / "train_grpo2.log")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
    panels = [
        ("Mean completion length (tokens)", "completions/mean_length", None, False),
        ("Fraction of zero-variance groups", "frac_reward_zero_std", (0, 1.05), True),
        ("Reward std within batch", "reward_std", (0, 0.8), True),
    ]
    runs = [("GRPO run 1 (group-std, reward v1, lr 2e-5)", r1, SERIES[0]),
            ("GRPO run 2 (Dr. GRPO, reward v2, KL 0.02, lr 1e-5)", r2, SERIES[1])]
    k = 5
    for ax, (title, key, ylim, smooth) in zip(axes, panels):
        for label, rows, color in runs:
            ys = [float(r[key]) for r in rows if key in r]
            xs = np.arange(1, len(ys) + 1)
            if smooth:
                sm = np.convolve(ys, np.ones(k) / k, mode="valid")
                ax.plot(xs, ys, color=color, lw=0.8, alpha=0.18)
                ax.plot(xs[k - 1:], sm, color=color, lw=2, solid_capstyle="round", label=label)
                ax.plot(xs[-1], sm[-1], marker="o", ms=6, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
            else:
                ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", label=label)
                ax.plot(xs[-1], ys[-1], marker="o", ms=6, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Optimizer step")
        if ylim:
            ax.set_ylim(*ylim)
        strip(ax)
    axes[0].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("GRPO training dynamics: run 1 drifts to long completions; run 2 stays stable", x=0.01,
                 ha="left", fontsize=10, fontweight="semibold")
    fig.text(0.01, -0.2, "Right two panels: 5-step moving average (bold) over raw per-step values (faint). Reward scales differ between runs (v1 vs v2) and are not compared directly.",
             fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_grpo_dynamics")


# ---- Table for the report ---------------------------------------------------------------
def write_table():
    rows = []
    for key, label in STAGES.items():
        s, h = summary(key), summary(HARD[key])
        eps, heps = episodes(key), episodes(HARD[key])
        p, lo, hi = wilson(sum(e["result"]["passed"] for e in eps), len(eps))
        hp, hlo, hhi = wilson(sum(e["result"]["passed"] for e in heps), len(heps))
        fmt_judge = lambda d: f"{d['mean_judge']:.2f}" if d.get("mean_judge") is not None else "-"  # noqa: E731
        rows.append({
            "stage": label.replace("\n", " "),
            "easy_pass": f"{100*p:.1f}% [{100*lo:.0f}, {100*hi:.0f}]",
            "easy_calls": f"{s['mean_calls']:.2f}",
            "easy_finish": f"{100*s['finished_cleanly_rate']:.0f}%",
            "easy_judge": fmt_judge(s),
            "hard_pass": f"{100*hp:.1f}% [{100*hlo:.0f}, {100*hhi:.0f}]",
            "hard_calls": f"{h['mean_calls']:.2f}",
            "hard_judge": fmt_judge(h),
        })
    hdr = "| Stage | Easy pass [95% CI] | Calls | Final msg | Judge | Hard pass [95% CI] | Calls | Judge |\n|---|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {r['stage']} | {r['easy_pass']} | {r['easy_calls']} | {r['easy_finish']} | {r['easy_judge']} | {r['hard_pass']} | {r['hard_calls']} | {r['hard_judge']} |\n" for r in rows)
    (OUT / "results_table.md").write_text(hdr + body, encoding="utf-8")
    print(hdr + body)


if __name__ == "__main__":
    check_inputs()
    fig_pass_rate()
    fig_secondary()
    fig_heatmap()
    fig_sft()
    fig_grpo()
    write_table()
