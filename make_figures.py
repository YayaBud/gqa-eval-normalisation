"""Figures for the measurement companion paper. Reads results/results.json.

Same discipline as the main study: no value is typed in here by hand. Two
figures, both regenerated from the canonical analysis:

  wsfig1  rank slope chart -- where each checkpoint sits under the stale
          normaliser and under the corrected one, on identical generations
  wsfig2  unweighted vs post-stratified difference for all 15 baseline pairs,
          with the disagreement quadrants marked
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "results.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_UP, C_DOWN, C_FLAT = "#1b7837", "#b2182b", "#777777"
# Text sizes at native figure scale. Figures are placed at \columnwidth, which
# enlarges them ~1.1-1.2x, so these print at about 8-9 pt (IEEE asks >= 8 pt).
FS_TXT, FS_AX = 7.5, 8.0   # labels/ticks/annotations, axis titles
plt.rcParams["mathtext.fontset"] = "stix"   # Times-like math, matching the text
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3 (PDF eXpress)

# rule A vs rule B gain/loss totals, from normdiff.py
_nd = json.load(open(os.path.join(HERE, "results", "normdiff.json"), encoding="utf-8"))
ND_TOTAL_GAINED, ND_TOTAL_LOST = _nd["total_gained"], _nd["total_lost"]


def short(name):
    return (name.replace("-Instruct", "").replace("InternVL3", "IVL3")
                .replace("Qwen2.5-VL", "Q2.5").replace("Qwen3-VL", "Q3")
                .replace("Qwen2-VL", "Q2"))


def _declutter(pairs, min_gap):
    """Nudge label y-positions apart, preserving order. Labels that collide are
    unreadable, and this chart has five checkpoints inside four points."""
    pairs = sorted(pairs, key=lambda t: t[0])
    ys = [p[0] for p in pairs]
    for _ in range(200):
        moved = False
        for i in range(len(ys) - 1):
            gap = ys[i + 1] - ys[i]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                ys[i] -= shift
                ys[i + 1] += shift
                moved = True
        if not moved:
            break
    return {p[1]: y for p, y in zip(pairs, ys)}


def wsfig1_ranks():
    NR = R["normalisation_ranks"]
    sysd = NR["systems"]
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    left = _declutter([(v["stale_normaliser"], k) for k, v in sysd.items()], 0.95 * FS_TXT / 5.4)
    right = _declutter([(v["fixed_normaliser"], k) for k, v in sysd.items()], 0.95 * FS_TXT / 5.4)
    for name, v in sysd.items():
        y0, y1 = v["stale_normaliser"], v["fixed_normaliser"]
        c = C_UP if v["rank_move"] > 0 else (C_DOWN if v["rank_move"] < 0 else C_FLAT)
        ax.plot([0, 1], [y0, y1], "-o", color=c, linewidth=1.0, markersize=2.6)
        ax.text(-0.05, left[name], f"{short(name)} ", ha="right", va="center",
                fontsize=FS_TXT, color=c)
        ax.text(1.05, right[name], f" {short(name)}", ha="left", va="center",
                fontsize=FS_TXT, color=c)
    ax.set_xlim(-0.70, 1.70)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rule A\n(rightmost)", "rule B\n(leftmost object)"],
                       fontsize=FS_TXT)
    ax.set_ylabel("exact match (%)", fontsize=FS_AX)
    ax.set_title(f"{NR['n_rank_changed']} of {NR['n_systems']} checkpoints change "
                 f"rank\n(identical stored generations)", fontsize=FS_AX, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)
    fig.savefig(os.path.join(OUT, "wsfig1_ranks.pdf")); fig.savefig(os.path.join(OUT, "wsfig1_ranks.png"), dpi=200)
    plt.close(fig)
    print("  wsfig1_ranks")


def wsfig2_pairs():
    BP = {k: v for k, v in R["baseline_pairs"].items() if not k.startswith("_")}
    fig, ax = plt.subplots(figsize=(3.4, 2.10))
    lim = 8.6
    ax.axhspan(-lim, 0, xmin=0.5, xmax=1.0, color="#f0c9c9", alpha=0.35, lw=0)
    ax.axhspan(0, lim, xmin=0.0, xmax=0.5, color="#f0c9c9", alpha=0.35, lw=0)
    ax.axhline(0, color="0.4", linewidth=0.6)
    ax.axvline(0, color="0.4", linewidth=0.6)
    ax.plot([-lim, lim], [-lim, lim], ":", color="0.6", linewidth=0.7)
    for k, v in BP.items():
        x, y = v["unweighted_delta"], v["poststrat_delta"]
        d = v["disagreement"]
        ax.scatter(x, y, s=15 if d else 9,
                   c=(C_DOWN if d == "sign_flip" else
                      ("#d98c00" if d == "verdict_change" else C_FLAT)),
                   marker="D" if d else "o", zorder=3, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("unweighted difference (points)", fontsize=FS_AX)
    ax.set_ylabel("post-stratified difference (points)", fontsize=FS_AX)
    s = R["baseline_pairs"]["_summary"]
    ax.set_title(f"{s['n_disagree']} of {s['n_pairs']} checkpoint pairs disagree\n"
                 f"({s['n_sign_flip']} reverse sign, "
                 f"{s['n_verdict_change']} change verdict)", fontsize=FS_AX, pad=4)
    ax.text(-lim + 0.4, lim - 0.6, "shaded: the two\nweightings disagree in sign",
            fontsize=FS_TXT, color="0.35", va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, labelsize=FS_TXT)
    fig.savefig(os.path.join(OUT, "wsfig2_pairs.pdf")); fig.savefig(os.path.join(OUT, "wsfig2_pairs.png"), dpi=200)
    plt.close(fig)
    print("  wsfig2_pairs")


def wsfig3_verbosity():
    """What predicts the size of the correction: answer phrasing, not capability."""
    p = os.path.join(HERE, "results", "verbosity.json")
    if not os.path.exists(p):
        print("  wsfig3 skipped (run verbosity.py first)")
        return
    V = json.load(open(p, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    for name, v in V["systems"].items():
        ax.scatter(v["raw_in_vocab_pct"], v["delta"], s=14, c="#2166ac",
                   zorder=3, linewidths=0)
        # Q2.5-7B sits level with IVL3-8B; at 8 pt their right-hand labels touch
        lft = short(name) == "Q2.5-7B"
        ax.annotate(short(name), (v["raw_in_vocab_pct"], v["delta"]),
                    textcoords="offset points", xytext=(-4, 3) if lft else (4, 3),
                    ha="right" if lft else "left", fontsize=FS_TXT, color="0.3")
    ax.set_xlabel("raw generations already in answer vocabulary (%)", fontsize=FS_AX)
    ax.set_ylabel("normalisation correction (points)", fontsize=FS_AX)
    ax.set_title(f"exposure tracks answer phrasing\n"
                 f"Spearman $\\rho={V['spearman_rho']:.2f}$, "
                 f"Pearson $r={V['pearson_r']:.2f}$ ($n={V['n']}$)",
                 fontsize=FS_AX, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, labelsize=FS_TXT)
    fig.savefig(os.path.join(OUT, "wsfig3_verbosity.pdf")); fig.savefig(os.path.join(OUT, "wsfig3_verbosity.png"), dpi=200)
    plt.close(fig)
    print("  wsfig3_verbosity")


def wsfig0_pipeline():
    """Methodology flow, with the two undocumented decisions drawn as forks.

    Row (a): one set of stored generations, two reduction rules, two
    leaderboards. Row (b): one set of per-type accuracies, two weightings,
    two verdicts. Everything right of the dashed line is a replay over saved
    text, so no model is loaded past that point.

    Column geometry is laid out explicitly on a 0-100 canvas. Every box edge
    is recorded and checked against RMAX at the end, because a row that sums
    past the canvas is clipped silently by savefig rather than raising.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    NR, S = R["normalisation_ranks"], R["baseline_pairs"]["_summary"]
    P = R["baseline_pairs"]["Qwen2.5-VL-3B vs Qwen3-VL-2B raw"]

    GRN, RED, GRY = "#1b7837", "#b2182b", "0.45"
    H, GAP, PAD, RMAX = 12.0, 2.6, 0.32, 98.0
    edges = []
    fig, ax = plt.subplots(figsize=(7.16, 2.60))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, top_s, bot_s="", ec=GRY, fc="white", lw=0.7, tc="0.15",
            bold=False):
        ax.add_patch(FancyBboxPatch((x, y - H / 2), w, H,
                                    boxstyle=f"round,pad={PAD}", facecolor=fc,
                                    edgecolor=ec, linewidth=lw, zorder=2))
        edges.append(x + w + PAD)
        if bot_s:
            ax.text(x + w / 2, y + 2.0, top_s, ha="center", va="center",
                    fontsize=5.3, color=tc, zorder=3,
                    fontweight="bold" if bold else "normal")
            ax.text(x + w / 2, y - 2.6, bot_s, ha="center", va="center",
                    fontsize=4.8, color="0.5", zorder=3)
        else:
            ax.text(x + w / 2, y, top_s, ha="center", va="center", fontsize=5.3,
                    color=tc, zorder=3, fontweight="bold" if bold else "normal")
        return x + w

    def arr(x1, y1, x2, y2, c="0.4"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), zorder=1,
                     arrowstyle="-|>", mutation_scale=6.0, linewidth=0.7,
                     color=c))

    def chain(x, y, items):
        for i, (w, t, s) in enumerate(items):
            x = box(x, y, w, t, s)
            if i < len(items) - 1:
                arr(x + 0.3, y, x + GAP - 0.3, y)
                x += GAP
        return x

    # ------------------------------------------------------------- row (a)
    yA, LANE = 75.0, 11.5
    ax.text(0.5, yA + 24, "(a)  decision 1: which token of a multi-token "
            "answer is scored", ha="left", fontsize=6.2, fontweight="bold",
            color="0.1")
    xa = chain(0.5, yA, [(11, "GQA", "val_balanced"),
                         (14, "two samples", "5,000 each"),
                         (15, "7 checkpoints", "single pass, bf16"),
                         (15.5, "stored generations", "raw text, kept")])
    ax.plot([xa + GAP / 2, xa + GAP / 2], [yA - 20, yA + 20], color="0.72",
            linewidth=0.7, linestyle=(0, (2.5, 2)))
    ax.text(xa + GAP / 2 + 0.8, yA + 21.5, "no model runs past here",
            fontsize=4.8, color="0.5", style="italic", ha="left", va="center")
    for sgn, lbl, sub, col, fill in ((+1, "rule A", "rightmost", RED, "#fdf0ef"),
                                     (-1, "rule B", "leftmost object", GRN,
                                      "#eef6ee")):
        y = yA + sgn * LANE
        arr(xa + 0.3, yA, xa + GAP - 0.3, y, c=col)
        xe = box(xa + GAP, y, 14.5, lbl, sub, ec=col, fc=fill, lw=0.9, tc=col,
                 bold=True)
        arr(xe + 0.3, y, xe + GAP - 0.3, y, c=col)
        box(xe + GAP, y, 13, "leaderboard", lbl[-1], ec=col, fc=fill, lw=0.9,
            tc=col)
    ax.text(0.5, yA - 21.5,
            f"the two leaderboards differ by {NR['delta_min']} to "
            f"{NR['delta_max']} points and disagree on the rank of "
            f"{NR['n_rank_changed']} of {NR['n_systems']} checkpoints; rule B "
            f"recovers {ND_TOTAL_GAINED:,} questions and loses "
            f"{ND_TOTAL_LOST}, so neither rule dominates",
            ha="left", fontsize=5.1, color=RED)

    # ------------------------------------------------------------- row (b)
    yB = 19.0
    ax.text(0.5, yB + 24, "(b)  decision 2: which distribution the per-type "
            "accuracies are averaged over", ha="left", fontsize=6.2,
            fontweight="bold", color="0.1")
    xb = chain(0.5, yB, [(22, "per-question correctness", "from either leaderboard"),
                         (19, "per-type accuracy", "five GQA types")])
    for sgn, lbl, sub, col, fill, val, verdict in (
            (+1, "equal weights", "balanced sample", GRN, "#eef6ee",
             f"{P['unweighted_delta']:+.2f}   z = {P['mcnemar_z']:+.2f}",
             "significant: 3B is better"),
            (-1, "GQA weights", "post-stratified", RED, "#fdf0ef",
             f"{P['poststrat_delta']:+.2f}   z = {P['poststrat_z']:+.2f}",
             "not significant: 3B is worse")):
        y = yB + sgn * LANE
        arr(xb + 0.3, yB, xb + GAP - 0.3, y, c=col)
        xe = box(xb + GAP, y, 18, lbl, sub, ec=col, fc=fill, lw=0.9, tc=col,
                 bold=True)
        arr(xe + 0.3, y, xe + GAP - 0.3, y, c=col)
        box(xe + GAP, y, 30, val, verdict, ec=col, fc=fill, lw=0.9, tc=col,
            bold=True)
    ax.text(0.5, yB - 21.5,
            f"shown for Qwen2.5-VL-3B against Qwen3-VL-2B, which leads on four "
            f"types and trails on query, 51.2% of GQA; over all pairs "
            f"{S['n_disagree']} of {S['n_pairs']} disagree and "
            f"{S['n_sign_flip']} reverse sign",
            ha="left", fontsize=5.1, color=RED)

    over = max(edges)
    assert over <= RMAX, f"row runs to {over:.2f} on a canvas ending at {RMAX}"

    fig.savefig(os.path.join(OUT, "wsfig0_pipeline.pdf")); fig.savefig(os.path.join(OUT, "wsfig0_pipeline.png"), dpi=200)
    plt.close(fig)
    print(f"  wsfig0_pipeline  (rightmost edge {over:.2f} / {RMAX})")


if __name__ == "__main__":
    print(f"regenerating from results/results.json (built {R['meta']['generated']})")
    # ponytail: Fig. 1 is the Word drawing made by extract_wordfig.py; the
    # matplotlib wsfig0_pipeline() below would overwrite it, so it is not run
    wsfig1_ranks()
    wsfig2_pairs()
    wsfig3_verbosity()
