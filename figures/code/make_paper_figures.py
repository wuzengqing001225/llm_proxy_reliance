#!/usr/bin/env python3
"""Build the four main-text figures from released summaries, without model calls.

Fig. 1B Verdict grid (Fig. 1A is a schematic drawn separately)
Fig. 2  Evidence tracking (four dose panels + slope forest)
Fig. 3  Representation and transfer (rendering control + real-record conditions)
Fig. 4  Names, examples and field structure (naming dot plot, error, reliance)

Palette: colorblind-safe categorical palette (Paul Tol), colour always paired
with marker shape or fill. Text is at least 7 pt at final size.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BLUE, CYAN, TEAL, ORANGE, RED, MAGENTA, GREY, BLACK = (
    "#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB", "#000000")
NEUTRAL, SOCIAL = BLUE, ORANGE
EXCESS, BELOW, COMPAT = RED, TEAL, "white"
FULL_W = 7.0          # in, double column
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Lato", "mathtext.it": "Lato:italic",
    "mathtext.bf": "Lato:bold", "mathtext.fallback": "stixsans",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "errorbar.capsize": 0,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
MODELS = [("sonnet", "Sonnet 4.5"), ("deepseek", "DeepSeek-V4-Flash"),
          ("qwen", "Qwen3.7-max"), ("gpt5.6-terra", "GPT-5.6 Terra")]


def letter(ax, s, x=-0.12, y=1.04):
    """Panel label: capital letter, upper-left (journal style)."""
    ax.text(x, y, s.upper(), transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left")


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ci(ax, x, y, lo, hi, color, marker="o", fill=True, size=4.2, horizontal=False,
       label=None, z=3, lw=1.0):
    face = color if fill else "white"
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=z - 1, solid_capstyle="butt")
        ax.plot(x, y, marker, ms=size, mfc=face, mec=color, mew=0.9, zorder=z, label=label)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, zorder=z - 1, solid_capstyle="butt")
        ax.plot(x, y, marker, ms=size, mfc=face, mec=color, mew=0.9, zorder=z, label=label)


# ------------------------------------------------------------------ data
def load(root):
    s = root / "data/summaries"
    g1 = pd.read_csv(s / "gate_summaries/expK_gate1_summary.csv")
    g2 = pd.read_csv(s / "gate_summaries/expK_gate2_summary.csv")
    frames = []
    for fn, key in (("deepseek-chat_summary.csv", "deepseek"), ("qwen3.7-max_summary.csv", "qwen"),
                    ("gpt-5.6-terra_summary.csv", "gpt5.6-terra")):
        f = pd.read_csv(s / fn)
        f = f[(f.k == 12) & (f.corner == "LH") & (f.beta.isna() | f.beta.eq(0.4))].copy()
        f["model"] = key
        frames.append(f)
    cross = pd.concat(frames, ignore_index=True)
    beta = pd.read_csv(s / "beta_four_model_summary.csv")
    slopes = pd.read_csv(s / "tracking_slopes.csv")
    ev = json.loads((s / "new_evidence.json").read_text())
    kt = pd.read_csv(s / "k_by_structure_summary.csv")
    ccp = root / "data/concurrent_checks/analysis/summary.json"
    cc = json.loads(ccp.read_text()) if ccp.exists() else None
    return g1, g2, cross, beta, slopes, ev, kt, cc


# ------------------------------------------------------------------ Fig 1
def verdict(lo, hi):
    return EXCESS if lo > 0 else (BELOW if hi < 0 else COMPAT)


def fig1(data, out):
    """Fig. 1B verdict grid. Fig. 1A is a schematic drawn separately."""
    g1, g2, cross, *_ = data
    conds = [(False, "neutral", "Zero information,\nneutral names"),
             (False, "social", "Zero information,\nsocial names"),
             (True, "neutral", "Informative,\nneutral names"),
             (True, "social", "Informative,\nsocial names")]
    rows = [("sonnet", 6, "Sonnet 4.5, $k$ = 6"), ("sonnet", 12, "Sonnet 4.5, $k$ = 12"),
            ("sonnet", 18, "Sonnet 4.5, $k$ = 18"), ("deepseek", 12, "DeepSeek-V4-Flash"),
            ("qwen", 12, "Qwen3.7-max"), ("gpt5.6-terra", 12, "GPT-5.6 Terra")]
    style = {"excess": (EXCESS, "o", True), "compatible": ("#555555", "o", False),
             "below": (BELOW, "s", True)}
    fig, axs = plt.subplots(1, 4, figsize=(4.35, 2.45), sharey=True,
                            gridspec_kw=dict(wspace=0.12))
    for ax, (info, sem, title) in zip(axs, conds):
        ax.axvline(0, color="black", lw=0.6, ls=(0, (3, 2)))
        for y, (key, k, _) in enumerate(rows):
            if key == "sonnet":
                r = g1[(g1.k == k) & (g1.informative == info) & (g1.semantic == sem)].iloc[0]
                v, lo, hi = r.excess_PSE_pp, r.excess_ci_lo, r.excess_ci_hi
            else:
                r = cross[(cross.model == key) & (cross.rule == "absent") & (cross.semantic == sem)
                          & (cross.informative == info)].iloc[0]
                v, lo, hi = r.excess_PSE_pp, r.PSE_lo - r.PSE_Bayes_pp, r.PSE_hi - r.PSE_Bayes_pp
            verdict = "excess" if lo > 0 else ("below" if hi < 0 else "compatible")
            col, mk, fill = style[verdict]
            ci(ax, v, y, lo, hi, col, mk, fill=fill, horizontal=True, size=3.8)
        ax.axhline(2.5, color="#DDDDDD", lw=0.6, zorder=0)
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xlim(-20, 32); ax.set_xticks([-10, 0, 10, 20, 30])
        ax.tick_params(axis="x", labelsize=6.5)
        ax.spines["left"].set_visible(ax is axs[0])
        if ax is not axs[0]:
            ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(range(len(rows)), [r[2] for r in rows]); axs[0].invert_yaxis()
    fig.supxlabel("PSE minus full-information reference (pp)", fontsize=7.5, y=-0.04, x=0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc=EXCESS, mec=EXCESS, ms=4, label="excess reliance"),
         Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", ms=4, label="compatible"),
         Line2D([], [], marker="s", ls="", mfc=BELOW, mec=BELOW, ms=4, label="below reference")]
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False,
               handletextpad=0.2, columnspacing=1.0)
    letter(axs[0], "b", x=-1.25, y=1.17)
    save(fig, out, "fig1b_verdicts")


# ------------------------------------------------------------------ Fig 2
def fig2(data, out):
    _, _, _, beta, slopes, *_ = data
    fig = plt.figure(figsize=(FULL_W, 2.55))
    w, h, x0, y0 = 0.16, 0.25, 0.065, 0.13
    axes = [fig.add_axes([x0 + c * (w + 0.035), y0 + (1 - r) * (h + 0.17), w, h])
            for r in range(2) for c in range(2)]
    for ax, (key, name) in zip(axes, MODELS):
        d = beta[(beta.model == key) & (beta.semantic == "neutral")].sort_values("beta")
        x = d.ideal_pp.to_numpy()
        ax.fill_between(x, x, d.bayes_ub_pp, color="#E8EEF3", lw=0, zorder=0)
        ax.plot([0, 24], [0, 24], color="#444444", lw=0.8)
        ax.plot(x, d.bayes_ub_pp, color="#888888", lw=0.8, ls=(0, (2, 1.5)))
        for sem, col, mk, ls in (("neutral", NEUTRAL, "o", "-"), ("social", SOCIAL, "s", "--")):
            e = beta[(beta.model == key) & (beta.semantic == sem)].sort_values("beta")
            ax.plot(e.ideal_pp, e.model_pse_pp, mk, ls=ls, color=col, ms=3.2, mfc=col if mk == "o" else "white",
                    mec=col, lw=0.9)
        ax.set_title(name, fontsize=7.5, pad=2)
        ax.set_xlim(-1.5, 25); ax.set_ylim(-2, 36); ax.set_xticks([0, 10, 20]); ax.set_yticks([0, 15, 30])
    for ax in axes[2:]:
        ax.set_xlabel("same-example\nreference (pp)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("model PSE (pp)")
    for ax in (axes[1], axes[3]):
        ax.set_yticklabels([])
    for ax in axes[:2]:
        ax.set_xticklabels([])
    letter(axes[0], "a", x=-0.45, y=1.55)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], marker="o", color=NEUTRAL, ms=3.2, label="neutral names"),
        Line2D([], [], marker="s", color=SOCIAL, mfc="white", ls="--", ms=3.2, label="social names"),
        Line2D([], [], color="#444444", lw=0.8, label="calibrated tracking"),
        Line2D([], [], color="#888888", lw=0.8, ls=(0, (2, 1.5)), label="full-information")],
        loc="lower left", bbox_to_anchor=(-0.05, 1.22), ncol=2, frameon=False, handlelength=1.6,
        columnspacing=0.9, handletextpad=0.4)
    # forest of slopes
    axf = fig.add_axes([0.62, 0.13, 0.375, 0.80])
    rows = []
    for key, name in MODELS:
        full = {"sonnet": "Sonnet 4.5", "deepseek": "DeepSeek-V4-Flash-0731",
                "qwen": "Qwen3.7-max", "gpt5.6-terra": "GPT-5.6 Terra"}[key]
        for lab in ("neutral", "social"):
            r = slopes[(slopes.model == full) & (slopes.labels == lab) & (slopes.protocol == "battery")].iloc[0]
            rows.append((f"{name.split('-V4')[0].split(' 4.5')[0].split('3.7')[0].split('-5.6')[0]}, {lab}", r, NEUTRAL if lab == "neutral" else SOCIAL,
                         "o" if lab == "neutral" else "s"))
    r = slopes[slopes.protocol == "repeated"].iloc[0]
    rows.append(("GPT, neutral, repeated", r, NEUTRAL, "D"))
    for nm in ("Pooled (3 models)", "Pooled (4 models)"):
        r = slopes[slopes.model == nm].iloc[0]
        rows.append((nm.replace("Pooled", "pooled") + ", 90%", r, BLACK, "D"))
    axf.axvspan(-0.15, 0.15, color="#EEEEEE", zorder=0, lw=0)
    for y, (lab, r, col, mk) in enumerate(rows):
        ci(axf, r.gamma, y, r.lo, r.hi, col, mk, fill=mk != "s", horizontal=True, size=3.6)
    axf.axvline(0, color="#777777", lw=0.6)
    axf.axvline(1, color="black", lw=0.8, ls=(0, (3, 2)))
    axf.axvline(0.5, color="#777777", lw=0.6, ls=":")
    axf.set_yticks(range(len(rows)), [r[0] for r in rows]); axf.invert_yaxis()
    axf.set_xlim(-0.35, 1.08); axf.set_xticks([0, 0.5, 1])
    axf.set_xlabel(r"tracking slope $\gamma$")
    axf.text(0.99, -0.75, "calibrated", fontsize=6.5, ha="right", va="bottom")
    axf.text(-0.14, -0.75, r"$\pm0.15$", fontsize=6.5, ha="left", va="bottom", color="#555555")
    axf.set_ylim(len(rows) - 0.5, -1.0)
    axf.axhline(7.5, color="#DDDDDD", lw=0.6); axf.axhline(8.5, color="#DDDDDD", lw=0.6)
    letter(axf, "b", x=-0.42, y=1.0)
    save(fig, out, "fig2_evidence_tracking")


# ------------------------------------------------------------------ Fig 3
def fig3(data, out):
    *_, ev, kt, cc = data
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(FULL_W, 2.15),
                                   gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.75))
    e = ev["encoding"]["learned"]
    rows = [("continuous numeric", e["numeric_reference"], BLUE, "o", True),
            ("ordered levels", e["ordinal"], TEAL, "o", True),
            ("arbitrary codes", e["nominal"], GREY if False else "#777777", "o", False),
            ("ordered minus codes", e["ordinal_minus_nominal"], BLACK, "D", True)]
    for y, (lab, d, col, mk, fill) in enumerate(rows):
        ci(axa, d["effect_pp"], y, d["lo_pp"], d["hi_pp"], col, mk, fill=fill, horizontal=True)
        axa.text(36.5, y, f"{d['effect_pp']:+.1f}".replace("-", "\u2212"), fontsize=6.6, va="center", ha="right")
    axa.axhline(2.5, color="#DDDDDD", lw=0.6)
    axa.axvline(0, color="#777777", lw=0.6)
    axa.set_yticks(range(len(rows)), [r[0] for r in rows]); axa.invert_yaxis()
    axa.set_xlim(-8, 37); axa.set_xlabel("directed proxy effect (pp)")
    axa.set_title("synthetic task, zero-information proxies", fontsize=7.5, pad=3, loc="left")
    letter(axa, "a", x=-0.62, y=1.02)
    spec = [("external_zero_primary_fixed_p1200", "primary, 1,200 pairs"),
            ("external_zero_random_target_fixed_p300", "random target, 300"),
            ("external_zero_primary_reverse_p300", "reversed coupling, 300"),
            ("external_zero_primary_seeded_p300", "permuted coupling, 300"),
            ("external_zero_gap_stratified_fixed_p1200", "gap-stratified, 150")]
    rows = [(lab, ev["external"][k]["all eligible pairs"], BLUE if i == 0 else "#555555")
            for i, (k, lab) in enumerate(spec)]
    if cc:
        rows += [("near tie, 600", cc["neartie"]["all"], ORANGE),
                 ("wide-gap control, 300", cc["wide"]["all"], ORANGE)]
    axb.axvspan(-5, 5, color="#F1F1F1", lw=0, zorder=0)
    for y, (lab, d, col) in enumerate(rows):
        ci(axb, d["effect_pp"], y, d["lo_pp"], d["hi_pp"], col, "o", horizontal=True)
    axb.axvline(0, color="#777777", lw=0.6)
    if cc:
        axb.axhline(len(spec) - 0.5, color="#DDDDDD", lw=0.6)
    axb.set_yticks(range(len(rows)), [r[0] for r in rows]); axb.invert_yaxis()
    axb.set_xlim(-6, 6); axb.set_xticks([-5, 0, 5])
    axb.set_xlabel("directed proxy effect (pp)")
    axb.set_title("real records, unordered categories", fontsize=7.5, pad=3, loc="left")
    letter(axb, "b", x=-0.62, y=1.02)
    save(fig, out, "fig3_representation_and_transfer")


# ------------------------------------------------------------------ Fig 4
def fig4(data, out):
    g1, g2, cross, *_ , kt, cc = data
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.2),
                            gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.42))
    ax = axs[0]
    for x, (key, name) in enumerate(MODELS):
        if key == "sonnet":
            n0 = g1[(g1.k == 12) & (~g1.informative) & (g1.semantic == "neutral")].iloc[0]
            s0 = g1[(g1.k == 12) & (~g1.informative) & (g1.semantic == "social")].iloc[0]
            s1 = g2[(g2.k == 12) & (~g2.informative) & (g2.semantic == "social")].iloc[0]
            pts = [(n0.PSE_pp, n0.PSE_ci_lo, n0.PSE_ci_hi), (s0.PSE_pp, s0.PSE_ci_lo, s0.PSE_ci_hi),
                   (s1.PSE_pp, s1.ci_lo, s1.ci_hi)]
        else:
            def g(rule, sem):
                r = cross[(cross.model == key) & (cross.rule == rule) & (cross.semantic == sem)
                          & (~cross.informative)].iloc[0]
                return (r.PSE_pp, r.PSE_lo, r.PSE_hi)
            pts = [g("absent", "neutral"), g("absent", "social"), g("learned", "social")]
        for dx, (v, lo, hi), col, mk, fill in zip((-0.22, 0, 0.22), pts, (NEUTRAL, SOCIAL, SOCIAL),
                                                  ("o", "s", "s"), (True, False, True)):
            ci(ax, x + dx, v, lo, hi, col, mk, fill=fill, size=3.8)
        ax.annotate("", xy=(x, pts[1][0]), xytext=(x - 0.22, pts[0][0]),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6))
        ax.annotate("", xy=(x + 0.22, pts[2][0]), xytext=(x, pts[1][0]),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6))
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xticks(range(4), ["Sonnet", "DeepSeek", "Qwen", "GPT"]); ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-19, 36); ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("zero-information PSE (pp)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=NEUTRAL, ms=3.8, label="neutral names"),
                       Line2D([], [], marker="s", ls="", mec=SOCIAL, mfc="white", ms=3.8, label="social names"),
                       Line2D([], [], marker="s", ls="", color=SOCIAL, ms=3.8,
                              label="social names, 80 examples")],
              loc="lower right", bbox_to_anchor=(1.02, -0.02), ncol=1, frameon=False,
              handletextpad=0.2, labelspacing=0.25)
    letter(ax, "a", x=-0.28, y=1.02)
    ax = axs[1]
    for table, col, mk, fill, lab in ((g1, "#777777", "o", False, "no examples"),
                                      (g2, BLUE, "o", True, "80 examples")):
        d = table[(~table.informative) & (table.semantic == "neutral")].sort_values("k")
        ax.plot(d.k, 100 * d.R_LLM, mk + "-", color=col, mfc=col if fill else "white", ms=3.6, label=lab)
    d = g2[(~g2.informative) & (g2.semantic == "neutral")].sort_values("k")
    ax.plot(d.k, 100 * d.R_Ideal, "^-", color=TEAL, ms=3.6, label="same-example learner")
    ax.set_xticks([6, 12, 18]); ax.set_xlabel("displayed fields $k$"); ax.set_ylabel("ranking error (%)")
    ax.set_ylim(-2, 45); ax.set_xlim(4.5, 19.5)
    ax.text(12, 39.5, "model, no examples", fontsize=6.6, color="#555555", ha="center")
    ax.text(13.2, 24.5, "model,\n80 examples", fontsize=6.6, color=BLUE, ha="left", va="center")
    ax.text(12, 5.5, "same-example learner", fontsize=6.6, color=TEAL, ha="center")
    letter(ax, "b", x=-0.3, y=1.02)
    ax = axs[2]
    lh = g2[(~g2.informative) & (g2.semantic == "neutral")].sort_values("k")
    hh = kt[(kt.corner == "HH") & (kt.rule == "learned") & (kt.semantic == "neutral")].sort_values("k")
    for d, lo, hi, col, mk, lab, dx in ((lh, "ci_lo", "ci_hi", BLUE, "o", "low dependence", -0.35),
                                        (hh, "lo", "hi", RED, "s", "high dependence", 0.35)):
        ax.plot(d.k + dx, d.PSE_pp, mk + "-", color=col, ms=3.6, label=lab, zorder=3)
        for _, r in d.iterrows():
            ax.plot([r.k + dx] * 2, [r[lo], r[hi]], color=col, lw=0.9)
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xticks([6, 12, 18]); ax.set_xlabel("displayed fields $k$"); ax.set_ylabel("PSE, 80 examples (pp)")
    ax.set_xlim(4.5, 19.5)
    ax.text(9.0, 25.0, "low task\ndependence", fontsize=6.6, color=BLUE, ha="center", va="bottom")
    ax.text(13.6, 8.0, "high task\ndependence", fontsize=6.6, color=RED, ha="left", va="bottom")
    letter(ax, "c", x=-0.3, y=1.02)
    save(fig, out, "fig4_names_examples_structure")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--outdir", type=Path)
    a = ap.parse_args()
    out = a.outdir or a.root / "figures/generated"
    out.mkdir(parents=True, exist_ok=True)
    data = load(a.root)
    fig1(data, out); fig2(data, out); fig3(data, out); fig4(data, out)
    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
