"""Why is the normaliser effect non-uniform across checkpoints?

Hypothesis: exact match penalises a model for how it phrases an answer, not
for whether it knows it. A checkpoint that answers "It is green." or
"a sulphur-crested cockatoo" loses credit that one answering "green" or
"parrot" keeps. If so, the size of the normalisation correction a checkpoint
receives should track how often its raw generation already lands in the GQA
answer vocabulary.

Measured on the stored generations -- no inference.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from gqa_answer_norm import normalize_answer, strip_direct_prefixes, build_vocab

RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROOT = os.path.join(RUNNER, "natural")
R = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                os.path.join("results", "results.json")), encoding="utf-8"))
IMP = R["normalisation_impact"]


def norm(x):
    return normalize_answer(strip_direct_prefixes(x or ""))


rows = {}
for name in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, name, f"pq_{name}.jsonl")
    if not os.path.exists(path):
        continue
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    vocab = build_vocab([norm(r["gt"]) for r in recs])
    raw_in = sum(norm(r.get("raw")) in vocab for r in recs)
    toks = [len((r.get("raw") or "").split()) for r in recs]
    rows[name] = {
        "n": len(recs),
        "raw_in_vocab_pct": round(100 * raw_in / len(recs), 2),
        "mean_raw_tokens": round(sum(toks) / len(toks), 2),
        "delta": IMP[name]["delta"],
    }

print(f"{'checkpoint':<26}{'raw in vocab':>13}{'mean tokens':>13}"
      f"{'norm. delta':>13}")
for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["delta"]):
    print(f"{k:<26}{v['raw_in_vocab_pct']:>12.2f}%{v['mean_raw_tokens']:>13.2f}"
          f"{v['delta']:>13.2f}")

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** .5
    sy = sum((b - my) ** 2 for b in ys) ** .5
    return cov / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0] * len(v)
    for pos, i in enumerate(order):
        rk[i] = pos + 1
    return rk


xs = [v["raw_in_vocab_pct"] for v in rows.values()]
ys = [v["delta"] for v in rows.values()]
n = len(xs)
r = pearson(xs, ys)
rho = pearson(rank(xs), rank(ys))
# leave-one-out: a rank correlation of -1 cannot come from one leverage point,
# but report the Pearson sensitivity anyway since that is what a reviewer asks
loo = [pearson([x for j, x in enumerate(xs) if j != i],
               [y for j, y in enumerate(ys) if j != i]) for i in range(n)]
print(f"\nPearson  r = {r:.3f}   Spearman rho = {rho:.3f}   (n={n} checkpoints)")
print(f"leave-one-out Pearson range: {min(loo):.3f} to {max(loo):.3f}")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "verbosity.json")
json.dump({"systems": rows, "pearson_r": round(r, 3), "spearman_rho": round(rho, 3),
           "loo_pearson_min": round(min(loo), 2), "loo_pearson_max": round(max(loo), 2),
           "n": n}, open(out, "w", encoding="utf-8"), indent=2)
print(f"wrote {out}")
