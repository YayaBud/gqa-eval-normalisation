"""Real worked examples of the two reduction rules, for the paper's table.

The table in the paper must show cases that actually occur in the stored
generations, not plausible-looking ones. This finds them: for each category
(rule B gains, rule A gains, the two agree) it reports genuine raw
generations together with the reference and what each rule scores.

Short raw strings are preferred only for column width; correctness of the
category is decided by the rules, not by the search.
"""
import json, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from gqa_answer_norm import (normalize_answer, strip_direct_prefixes,
                             build_vocab, snap_to_vocab)

RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROOT = os.path.join(RUNNER, "natural")


def norm(x):
    return normalize_answer(strip_direct_prefixes(x or ""))


buckets = collections.defaultdict(list)
for name in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, name, f"pq_{name}.jsonl")
    if not os.path.exists(p):
        continue
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    gt = {r["qid"]: norm(r["gt"]) for r in recs}
    vocab = build_vocab(list(gt.values()))
    for r in recs:
        g = gt[r["qid"]]
        raw = (r.get("raw") or "").strip()
        a = norm(r.get("final"))                       # rule A: rightmost
        b = snap_to_vocab(norm(r.get("raw")), vocab)   # rule B: leftmost non-attr
        if not raw or a == b:
            key = "agree"
        elif b == g and a != g:
            key = "B gains"
        elif a == g and b != g:
            key = "A gains"
        else:
            key = "both wrong"
        buckets[key].append((len(raw), raw, g, a, b, name))

for key in ("B gains", "A gains", "agree", "both wrong"):
    rows = sorted(buckets[key])
    print("=" * 96)
    print(f"{key}   ({len(rows)} cases)")
    print("=" * 96)
    print(f"  {'raw generation':<34}{'reference':<14}{'rule A':<14}{'rule B':<14}")
    seen = set()
    shown = 0
    for _, raw, g, a, b, name in rows:
        if raw.lower() in seen or len(raw) < 6:
            continue
        seen.add(raw.lower())
        print(f"  {raw[:32]:<34}{g[:12]:<14}{a[:12]:<14}{b[:12]:<14}{name}")
        shown += 1
        if shown >= 8:
            break
    print()
