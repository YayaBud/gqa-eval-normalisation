"""Characterise the two normaliser versions precisely, for all checkpoints.

Both versions strip conversational prefixes and normalise lexically. They
differ in one rule: how a multi-token answer is reduced to a single scored
token. The earlier version takes the RIGHTMOST vocabulary match; the later
takes the LEFTMOST match that is not an attribute token (colour, material,
size, position, count), falling back to rightmost when only attribute tokens
match. Neither rule is obviously wrong, and each fails where the other
succeeds -- which is the point.

Emits normdiff.json for the paper and its verifier.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from gqa_answer_norm import (normalize_answer, strip_direct_prefixes,
                             build_vocab, snap_to_vocab)

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(RUNNER, "natural")


def norm(x):
    return normalize_answer(strip_direct_prefixes(x or ""))


out = {}
for name in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, name, f"pq_{name}.jsonl")
    if not os.path.exists(p):
        continue
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    gt = {r["qid"]: norm(r["gt"]) for r in recs}
    vocab = build_vocab(list(gt.values()))
    gained = lost = agree = 0
    for r in recs:
        g = gt[r["qid"]]
        s_txt = norm(r.get("final"))
        f_txt = snap_to_vocab(norm(r.get("raw")), vocab)
        agree += (s_txt == f_txt)
        s, f = s_txt == g, f_txt == g
        gained += (f and not s)
        lost += (s and not f)
    n = len(recs)
    out[name] = {"n": n, "gained": gained, "lost": lost, "net": gained - lost,
                 "agree_pct": round(100 * agree / n, 1),
                 "disagree_pct": round(100 * (n - agree) / n, 1)}

print(f"{'checkpoint':<26}{'gained':>8}{'lost':>7}{'net':>7}{'strings differ':>16}")
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["net"]):
    print(f"{k:<26}{v['gained']:>8}{v['lost']:>7}{v['net']:>7}"
          f"{v['disagree_pct']:>15.1f}%")

tot_g = sum(v["gained"] for v in out.values())
tot_l = sum(v["lost"] for v in out.values())
print(f"\nacross all {len(out)} checkpoints: {tot_g} gained, {tot_l} lost")
print(f"every checkpoint loses some: {all(v['lost'] > 0 for v in out.values())}")
print(f"disagreement range: {min(v['disagree_pct'] for v in out.values()):.1f}%"
      f" - {max(v['disagree_pct'] for v in out.values()):.1f}%")

json.dump({"systems": out, "total_gained": tot_g, "total_lost": tot_l,
           "all_lose_some": all(v["lost"] > 0 for v in out.values()),
           "disagree_min": min(v["disagree_pct"] for v in out.values()),
           "disagree_max": max(v["disagree_pct"] for v in out.values())},
          open(os.path.join(HERE, "results", "normdiff.json"), "w", encoding="utf-8"),
          indent=2)
print("wrote normdiff.json")
