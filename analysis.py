"""Every number in the INSPECT 2026 paper, regenerated from the stored generations.

    python analysis.py            -> results/results.json

No inference and no GPU: each figure is a replay over the saved per-question
outputs in data/. Two independent 5,000-question draws from GQA are used:

  data/natural/   the natural type mix (seed 42, test subset), 7 checkpoints
  data/balanced/  1,000 questions per structural type, 6 checkpoints

The functions below are extracted unchanged from the main study's canonical
analysis script, restricted to the public single-pass checkpoints.
"""
import collections, itertools, json, math, os, random, sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stdlib_boot  # exact multinomial bootstrap, stdlib only
from gqa_answer_norm import (normalize_answer, strip_direct_prefixes,
                             build_vocab, snap_to_vocab)

DATA = os.path.join(HERE, "data")

# True GQA val_balanced structural-type proportions -> post-stratification.
W = {"choose": .1270, "compare": .0314, "logical": .1202,
     "query": .5124, "verify": .2090}
TYPES = ["choose", "compare", "logical", "query", "verify"]

BALANCED = [
    ("InternVL3-14B",   "balanced/InternVL3-14B/pq_InternVL3-14B.jsonl"),
    ("InternVL3-8B",    "balanced/InternVL3-8B/pq_InternVL3-8B.jsonl"),
    ("InternVL3-2B",    "balanced/InternVL3-2B/pq_InternVL3-2B.jsonl"),
    ("Qwen2-VL-2B",     "balanced/Qwen2-VL-2B-Instruct/pq_Qwen2-VL-2B-Instruct.jsonl"),
    ("Qwen2.5-VL-3B",   "balanced/Qwen2.5-VL-3B-Instruct/pq_Qwen2.5-VL-3B-Instruct.jsonl"),
    ("Qwen3-VL-2B raw", "balanced/Qwen3-VL-2B-Instruct/pq_Qwen3-VL-2B-Instruct.jsonl"),
]
BASELINE_SYSTEMS = [n for n, _ in BALANCED]


def norm(x):
    return normalize_answer(strip_direct_prefixes(str(x or "")))


def read(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def balanced_table():
    """Per-type and post-stratified exact match on the balanced draw.

    Every checkpoint is rescored from its raw generation with one shared
    vocabulary snap, so no system is scored by a different normaliser.
    """
    gt, preds = {}, {}
    for name, rel in BALANCED:
        p = {}
        for r in read(os.path.join(DATA, rel)):
            gt.setdefault(r["qid"], (norm(r["gt"]), r["type"]))
            p[r["qid"]] = norm(r.get("raw"))
        preds[name] = p
    vocab = build_vocab([g for g, _ in gt.values()])
    systems = {}
    for name, p in preds.items():
        p = {q: snap_to_vocab(v, vocab) for q, v in p.items()}
        preds[name] = p
        per = {t: [0, 0] for t in TYPES}
        for q, a in p.items():
            g, t = gt[q]
            per[t][0] += 1
            per[t][1] += (a == g)
        acc = {t: 100 * per[t][1] / per[t][0] for t in TYPES if per[t][0]}
        wsum = sum(W[t] for t in acc)
        systems[name] = {
            "n": len(p),
            "per_type": {t: {"n": per[t][0], "correct": per[t][1],
                             "em": round(acc[t], 2)} for t in acc},
            "macro": round(sum(acc.values()) / len(acc), 2),
            "natural": round(sum(W[t] * acc[t] for t in acc) / wsum, 2),
        }
    return gt, preds, systems


def mcnemar(gt, A, B):
    common = sorted(set(A) & set(B))
    b = sum(1 for q in common if A[q] == gt[q][0] and B[q] != gt[q][0])
    c = sum(1 for q in common if A[q] != gt[q][0] and B[q] == gt[q][0])
    z = (b - c) / math.sqrt(b + c) if (b + c) else 0.0
    return {"n": len(common), "a_only": b, "b_only": c, "net": b - c,
            "z": round(z, 2), "significant": abs(z) > 1.96}


def poststrat_test(gt, A, B, boot=20000, seed=0):
    """Paired test of the POST-STRATIFIED difference. McNemar does not test it.

    On a type-balanced draw McNemar tests the *macro* difference. Per question
    i of type t let d_i = 1[A correct] - 1[B correct]. Then
    D = sum_t w_t * mean_t(d) and Var(D) = sum_t w_t^2 * s_t^2 / n_t, with the
    weights taken as known constants (measured on the full val_balanced split).
    A stratified paired bootstrap, done in closed form because d has three
    support points, is reported alongside.
    """
    common = sorted(set(A) & set(B))
    by_type = collections.defaultdict(list)
    for q in common:
        by_type[gt[q][1]].append((A[q] == gt[q][0]) - (B[q] == gt[q][0]))
    rng = random.Random(seed)
    D, tvar, boots = 0.0, 0.0, None
    for t in TYPES:
        d = by_type.get(t, [])
        n = len(d)
        if not n:
            continue
        D += W[t] * (sum(d) / n)
        tvar += W[t] ** 2 * stdlib_boot.var(d, ddof=1) / n
        p = [d.count(-1) / n, d.count(0) / n, d.count(1) / n]
        c = stdlib_boot.multinomial(rng, n, p, boot)
        contrib = [W[t] * (x[2] - x[0]) / n for x in c]
        boots = contrib if boots is None else [a + b for a, b in zip(boots, contrib)]
    se = math.sqrt(tvar)
    z = D / se if se else 0.0
    lo, hi = stdlib_boot.percentile(boots or [0.0], [2.5, 97.5])
    return {"delta_pts": round(100 * D, 2), "se_pts": round(100 * se, 2),
            "z": round(z, 2), "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)],
            "significant": bool(abs(z) > 1.96)}


def baseline_pairs(gt, preds, boot=20000):
    """Every pair of checkpoints, unweighted (McNemar) vs post-stratified.

    A pair is flagged when the two analyses disagree: the difference reverses
    sign, or one is significant and the other is not.
    """
    out = {}
    for a, b in itertools.combinations(BASELINE_SYSTEMS, 2):
        m = mcnemar(gt, preds[a], preds[b])
        w = poststrat_test(gt, preds[a], preds[b], boot=boot)
        unw = round(100 * m["net"] / m["n"], 2)
        if unw * w["delta_pts"] < 0:
            flag = "sign_flip"
        elif m["significant"] != w["significant"]:
            flag = "verdict_change"
        else:
            flag = None
        out[f"{a} vs {b}"] = {
            "unweighted_delta": unw, "mcnemar_z": m["z"],
            "unweighted_significant": m["significant"],
            "poststrat_delta": w["delta_pts"], "poststrat_z": w["z"],
            "poststrat_significant": w["significant"],
            "poststrat_ci95": w["ci95_pts"], "disagreement": flag,
        }
    flags = [v["disagreement"] for v in out.values()]
    out["_summary"] = {
        "n_pairs": len(flags),
        "n_disagree": sum(1 for f in flags if f),
        "n_sign_flip": flags.count("sign_flip"),
        "n_verdict_change": flags.count("verdict_change"),
    }
    return out


def normalisation_impact():
    """Same stored generations, two normaliser versions (natural draw).

    stale = the answer as scored at generation time by the earlier normaliser
    (rule A, rightmost vocabulary match); fixed = the raw generation rescored
    with the current one (rule B). See gqa_answer_norm.py.
    """
    out = {}
    root = os.path.join(DATA, "natural")
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, f"pq_{name}.jsonl")
        if not os.path.exists(path):
            continue
        rows = list(read(path))
        gt = {r["qid"]: norm(r["gt"]) for r in rows}
        vocab = build_vocab(list(gt.values()))
        stale = sum(norm(r.get("final")) == gt[r["qid"]] for r in rows)
        fixed = sum(snap_to_vocab(norm(r.get("raw")), vocab) == gt[r["qid"]]
                    for r in rows)
        n = len(rows)
        out[name] = {"n": n,
                     "stale_normaliser": round(100 * stale / n, 2),
                     "fixed_normaliser": round(100 * fixed / n, 2),
                     "delta": round(100 * (fixed - stale) / n, 2)}
    return out


def normalisation_ranks(imp):
    st = sorted(imp, key=lambda k: -imp[k]["stale_normaliser"])
    fx = sorted(imp, key=lambda k: -imp[k]["fixed_normaliser"])
    rs = {k: i + 1 for i, k in enumerate(st)}
    rf = {k: i + 1 for i, k in enumerate(fx)}
    return {"systems": {k: {**imp[k], "rank_stale": rs[k], "rank_fixed": rf[k],
                            "rank_move": rs[k] - rf[k]} for k in imp},
            "order_stale": st, "order_fixed": fx,
            "n_systems": len(imp),
            "n_rank_changed": sum(1 for k in imp if rs[k] != rf[k]),
            "delta_min": round(min(v["delta"] for v in imp.values()), 2),
            "delta_max": round(max(v["delta"] for v in imp.values()), 2)}


def main():
    gt, preds, systems = balanced_table()
    imp = normalisation_impact()
    R = {
        "meta": {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "post_strat_weights": W},
        "balanced": systems,
        "baseline_pairs": baseline_pairs(gt, preds),
        "normalisation_impact": imp,
        "normalisation_ranks": normalisation_ranks(imp),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    nr, s = R["normalisation_ranks"], R["baseline_pairs"]["_summary"]
    print(f"normaliser shift {nr['delta_min']} to {nr['delta_max']} pts; "
          f"{nr['n_rank_changed']} of {nr['n_systems']} change rank")
    print(f"{s['n_pairs']} pairs: {s['n_disagree']} disagree, "
          f"{s['n_sign_flip']} sign flips, {s['n_verdict_change']} verdict changes")
    print("->", out)


if __name__ == "__main__":
    main()
