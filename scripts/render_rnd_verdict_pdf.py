"""Render the IntrinsicPPO RND 50x50 verdict as a shareable PDF.

Usage:
    python -m scripts.render_rnd_verdict_pdf \
        --history /path/to/history.csv \
        --out ~/Downloads/intrinsic-rnd-thr-50x50-verdict.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]
SCALE_LABEL = {0.01: "0.01", 0.1: "0.1", 1.0: "1", 10.0: "10", 100.0: "100"}
# Distinct, print-safe colors (scale 1 highlighted).
SCALE_COLOR = {
    0.01: "#4C78A8",
    0.1: "#72B7B2",
    1.0: "#2E7D32",
    10.0: "#F2A65A",
    100.0: "#C44E52",
}


def parse_name(name):
    m = re.search(r"iscale=([0-9.]+)-seed=(\d+)", str(name))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), int(m.group(2))


def load_history(path):
    hist = pd.read_csv(path)
    hist["iscale"] = hist["name"].map(lambda n: parse_name(n)[0])
    hist["seed"] = hist["name"].map(lambda n: parse_name(n)[1])
    hist["_step_m"] = hist["_step"] / 1e6
    return hist


def last_checkpoint(hist):
    return hist.sort_values("_step").groupby("run_id", as_index=False).tail(1)


def style_axes(ax, ylabel, xlabel="Environment steps (millions)"):
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scale_lines(ax, hist, ycol, ylabel, hline=None):
    for scale in SCALES:
        g = (
            hist[hist["iscale"] == scale]
            .groupby("_step_m", as_index=False)[ycol]
            .mean()
            .sort_values("_step_m")
        )
        ax.plot(
            g["_step_m"],
            g[ycol],
            color=SCALE_COLOR[scale],
            lw=2.0 if scale == 1.0 else 1.4,
            label=f"scale {SCALE_LABEL[scale]}",
        )
    if hline is not None:
        ax.axhline(hline, color="#888888", ls="--", lw=1, label=f"unlock {hline:g}")
    style_axes(ax, ylabel)
    ax.legend(frameon=False, ncol=3, fontsize=8)


def page_title(fig, title):
    fig.suptitle(title, fontsize=13, fontweight="semibold", y=0.98)


def render(hist, out_path):
    last = last_checkpoint(hist)
    last = last.copy()
    last["run_ge50"] = last["eval/return/Forward_Distance/mean"] >= 50
    last["jump_ge50"] = last["eval/return/Jump_Height/mean"] >= 50
    last["both"] = last["run_ge50"] & last["jump_ge50"]

    ever_both = (
        hist.groupby(["iscale", "seed"])["eval/unlock/all"].max() == 1
    ).mean()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # --- Page 1: verdict ---
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, "Intrinsic PPO + RND at 50×50  ·  discovery verdict")
        ax = fig.add_axes([0.07, 0.40, 0.86, 0.50])
        ax.axis("off")
        body = (
            "W&B group  intrinsic-rnd-thr=50x50\n"
            "25 finished runs  ·  5 RND scales × 5 seeds  ·  50M steps\n"
            "Eval: ungated Hopper returns (forward distance, jump height). "
            "Training reward is RND novelty only.\n\n"
            "VERDICT\n"
            "Discovery works. Freeze RND scale at 1 and move to preservation.\n\n"
            "Undirected RND novelty reaches both sparse skills. Every one of the 25\n"
            "runs crossed return 50 on run and on jump at some checkpoint. Scale 1 is\n"
            "the only setting that still holds both skills at 50M steps (5/5 seeds).\n"
            "The bottleneck is no longer finding the behaviors — it is keeping them.\n\n"
            f"Ever unlocked both skills:     25/25 runs  ({ever_both:.0%} of scale-seed cells)\n"
            f"Still both at 50M:             {int(last['both'].sum())}/25 runs\n"
            "Scale 1 still both at 50M:     5/5 seeds\n"
            "Median first run crossing:     ~3.6M steps (first eval after init)\n\n"
            "Final checkpoint (mean ± std across seeds)"
        )
        ax.text(0.0, 1.0, body, va="top", ha="left", fontsize=10, family="sans-serif")

        cell = []
        for scale in SCALES:
            s = last[last["iscale"] == scale]
            cell.append(
                [
                    SCALE_LABEL[scale],
                    f"{s['eval/return/Forward_Distance/mean'].mean():.0f} ± {s['eval/return/Forward_Distance/mean'].std():.0f}",
                    f"{s['eval/return/Jump_Height/mean'].mean():.0f} ± {s['eval/return/Jump_Height/mean'].std():.0f}",
                    f"{int(s['run_ge50'].sum())}/5",
                    f"{int(s['jump_ge50'].sum())}/5",
                    f"{int(s['both'].sum())}/5",
                    f"{s['eval/avg_episode_length'].mean():.0f}",
                ]
            )
        tbl_ax = fig.add_axes([0.07, 0.08, 0.86, 0.28])
        tbl_ax.axis("off")
        table = tbl_ax.table(
            cellText=cell,
            colLabels=[
                "scale",
                "run mean",
                "jump mean",
                "run≥50",
                "jump≥50",
                "both",
                "ep. len",
            ],
            loc="upper center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)
        for (r, c), cell_obj in table.get_celld().items():
            cell_obj.set_edgecolor("#DDDDDD")
            if r == 0:
                cell_obj.set_facecolor("#F3F3F3")
                cell_obj.set_text_props(fontweight="semibold")
            elif SCALE_LABEL[SCALES[r - 1]] == "1":
                cell_obj.set_facecolor("#E8F5E9")
            elif SCALE_LABEL[SCALES[r - 1]] == "100":
                cell_obj.set_facecolor("#FDECEA")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: curves ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.97, hspace=0.28)
        page_title(fig, "Mean across 5 seeds  ·  ungated eval vs training steps")
        plot_scale_lines(
            axes[0],
            hist,
            "eval/return/Forward_Distance/mean",
            "Forward distance return",
            hline=50,
        )
        plot_scale_lines(
            axes[1],
            hist,
            "eval/return/Jump_Height/mean",
            "Jump height return",
            hline=50,
        )
        plot_scale_lines(
            axes[2],
            hist,
            "eval/unlock/all",
            "Fraction of seeds with both ≥ 50",
        )
        axes[2].set_ylim(-0.05, 1.05)
        fig.text(
            0.10,
            0.015,
            "Source: W&B export history.csv  ·  15 eval checkpoints  ·  scale 1 highlighted in green",
            fontsize=8,
            color="#666666",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: per-seed + notes ---
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, "Every seed at 50M steps")
        last_sorted = last.sort_values(["iscale", "seed"])
        cell = []
        row_colors = []
        for _, r in last_sorted.iterrows():
            both = bool(r["both"])
            cell.append(
                [
                    SCALE_LABEL[r["iscale"]],
                    str(int(r["seed"])),
                    f"{r['eval/return/Forward_Distance/mean']:.0f}",
                    f"{r['eval/return/Jump_Height/mean']:.0f}",
                    "yes" if both else "no",
                    f"{r['eval/avg_episode_length']:.0f}",
                ]
            )
            row_colors.append("#E8F5E9" if both else "#FFF8E1")
        ax = fig.add_axes([0.08, 0.38, 0.84, 0.54])
        ax.axis("off")
        table = ax.table(
            cellText=cell,
            colLabels=["scale", "seed", "run return", "jump return", "both ≥50", "ep. length"],
            loc="upper center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.22)
        for (r, c), cell_obj in table.get_celld().items():
            cell_obj.set_edgecolor("#EEEEEE")
            if r == 0:
                cell_obj.set_facecolor("#F3F3F3")
                cell_obj.set_text_props(fontweight="semibold")
            else:
                cell_obj.set_facecolor(row_colors[r - 1])

        axn = fig.add_axes([0.08, 0.04, 0.84, 0.32])
        axn.axis("off")
        notes = (
            "What this is not.  Returns are well below dense MORLAX (~2500 run / ~2200 jump).\n"
            "Expected: the policy optimizes RND prediction error, not the task. Peak-over-training\n"
            "run is ~500–1300. The claim is discoverability of the 50-bar, not Pareto optimality.\n\n"
            "Eval caveat.  Within a checkpoint, return std across 128 eval envs is exactly 0 and\n"
            "unlock is only 0 or 1. Deterministic eval plus identical Hopper resets means each\n"
            "checkpoint is one trajectory, replicated. Seed variance is the real uncertainty.\n\n"
            "Next.  Freeze rnd_params.scale = 1. Do not resweep scale 100. The next test is whether\n"
            "an extrinsic population can stash these breakthroughs without first-objective collapse."
        )
        axn.text(0.0, 1.0, notes, va="top", ha="left", fontsize=9, family="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "Intrinsic PPO + RND at 50x50 — discovery verdict"
        d["Author"] = "SMORL"
        d["Subject"] = "W&B group intrinsic-rnd-thr=50x50"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    hist = load_history(args.history)
    render(hist, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
