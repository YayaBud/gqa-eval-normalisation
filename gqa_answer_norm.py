"""
gqa_answer_norm.py — GQA answer normalization + vocabulary snapping
====================================================================
Fixes the "left spoon" vs "spoon" class of misses: the model is right
but verbose, and exact-match punishes it.

Shared by vlm_inference.py, gqa_runner.py and evaluate_metrics.py.

GQA is effectively classification over a fixed answer vocabulary
(~1.8k answers), so snapping the free-text prediction onto that
vocabulary is standard practice, not cheating: the vocabulary comes
from the dataset as a whole, never from the current question's GT.

Changes vs original:
  - strip_direct_prefixes() strips model verbosity from short direct answers
  - _levenshtein() + fuzzy fallback in snap_to_vocab catches plurals/typos
"""

import re

ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s-]")

# Attribute tokens (colour / material / size / position / count). These ARE
# valid GQA answers on their own, but when a model lists an object noun
# alongside them ("van white metal"), the object is the answer and the
# attributes are noise. snap_to_vocab demotes these ONLY when a non-attribute
# object token co-occurs, so a standalone "green" or "left" is left untouched.
_ATTRIBUTE_TOKENS = {
    # colours
    "red", "blue", "green", "yellow", "white", "black", "brown", "gray",
    "grey", "orange", "pink", "purple", "gold", "silver", "tan", "beige",
    "dark", "light", "bright",
    # materials
    "wood", "wooden", "metal", "metallic", "plastic", "glass", "leather",
    "stone", "fabric", "cloth", "ceramic", "rubber", "paper", "cardboard",
    "concrete", "steel", "cotton", "denim",
    # sizes / age
    "large", "small", "tall", "short", "big", "little", "long", "old", "new",
    "young", "tiny", "huge", "wide", "narrow", "thick", "thin",
    # positions
    "left", "right", "top", "bottom", "center", "centre", "middle", "front",
    "back", "side", "upper", "lower",
    # counts
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
}

_FILLER_PREFIXES = (
    "it is ", "it's ", "this is ", "that is ", "there is ", "there are ",
    "the answer is ", "answer: ", "they are ", "i see ",
)

# Verbosity prefixes that appear even in short "direct" VQA answers
_DIRECT_PREFIXES = (
    "yes, ", "no, ", "yes it is ", "no it is not ", "yes it's ",
    "no it's not ", "the answer is ", "i think ", "i believe ",
    "it appears to be ", "it looks like ", "it seems like ",
    "there is a ", "there is an ", "this is a ", "this is an ",
)


# Yes/No only counts as a verdict when followed by punctuation or a
# sentence continuation — NOT by an arbitrary noun. Otherwise real
# answers like "no parking" collapse to "no".
_YESNO_RE = re.compile(
    r"^(yes|no)(?:[,.!]|\s+(?:it|its|there|this|that|he|she|they|is|was|the)\b)"
)


def strip_direct_prefixes(text: str) -> str:
    """Strip common model verbosity from short direct VQA answers.

    Run this BEFORE normalize_answer() on raw direct-pass output.

    Examples:
        'Yes, it is blue'          -> 'yes'
        'No, there is none'        -> 'no'
        'no parking'               -> 'no parking'   (preserved!)
        'it appears to be red'     -> 'red'
        'I think the dog is brown' -> 'the dog is brown'
    """
    t = str(text).lower().strip()

    # Yes/No verdicts — extract the token and stop immediately
    m = _YESNO_RE.match(t)
    if m:
        return m.group(1)

    # Strip remaining filler prefixes
    changed = True
    while changed:
        changed = False
        for p in _DIRECT_PREFIXES:
            if t.startswith(p):
                t = t[len(p):]
                changed = True
    return t.strip()


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, articles, and filler prefixes."""
    t = str(text).lower().strip()
    changed = True
    while changed:
        changed = False
        for p in _FILLER_PREFIXES:
            if t.startswith(p):
                t = t[len(p):]
                changed = True
    t = _PUNCT.sub(" ", t)
    toks = [w for w in t.split() if w not in ARTICLES]
    return " ".join(toks)


def build_vocab(answers) -> set:
    """Build the answer vocabulary from an iterable of GT answers."""
    return {normalize_answer(a) for a in answers if str(a).strip()}


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,        # deletion
                curr[j] + 1,            # insertion
                prev[j] + (ca != cb),   # substitution
            ))
        prev = curr
    return prev[-1]


def snap_to_vocab(pred: str, vocab: set | None) -> str:
    """
    Snap a verbose prediction onto the answer vocabulary.

      'left spoon'       -> 'spoon'        (head noun in vocab)
      'light blue color' -> 'light blue'   (longest vocab n-gram)
      'yes, it is'       -> 'yes'
      'on the left'      -> 'left'
      'bicycles'         -> 'bicycle'      (fuzzy: 1 edit, NEW)

    Priority:
      1. Exact match after normalization
      2. Yes/No early exit
      3. Longest n-gram match in vocab  (rightmost wins for head-noun)
      4. Fuzzy edit-distance ≤ 2        (single-token preds only, NEW)

    Falls back to the normalized prediction if nothing matches.
    """
    p = normalize_answer(pred)
    if not p or not vocab or p in vocab:
        return p
    toks = p.split()

    # Early exit for yes/no
    if toks[0] in ("yes", "no"):
        return toks[0]

    # ── Multi-word n-gram match (size >= 2): longest, rightmost wins ──
    # Handles "light blue color" -> "light blue" before any single token.
    for size in range(len(toks), 1, -1):
        for start in range(len(toks) - size, -1, -1):
            cand = " ".join(toks[start:start + size])
            if cand in vocab:
                return cand

    # ── Single-token match: object noun beats attribute tokens ──
    # A model asked for "object name, colour, material or number" may answer
    # with ALL of them ("van white metal"). The object noun is the answer;
    # prefer the leftmost non-attribute vocab token. If only attribute tokens
    # match, fall back to the old rightmost-wins behaviour so a standalone
    # colour/position answer is unchanged.
    single_matches = [t for t in toks if t in vocab]
    if single_matches:
        objects = [t for t in single_matches if t not in _ATTRIBUTE_TOKENS]
        if objects:
            return objects[0]              # leading object noun
        return single_matches[-1]          # attribute-only: preserve old pick

    # ── Fuzzy fallback (single-word preds only) ──
    if len(toks) == 1:
        # Deterministic plural/singular fast path first
        if p.endswith("s") and p[:-1] in vocab:
            return p[:-1]
        if p + "s" in vocab:
            return p + "s"
        # Edit distance guarded three ways:
        #   - first letter must match (kills "pup" → "cup")
        #   - threshold scales with length (short words flip too easily)
        #   - sorted(vocab) so ties resolve deterministically across runs
        max_d = 1 if len(p) <= 4 else 2
        best_word, best_dist = None, max_d + 1
        for v in sorted(vocab):
            if not v or v[0] != p[0] or abs(len(v) - len(p)) > max_d:
                continue
            d = _levenshtein(p, v)
            if d < best_dist:
                best_word, best_dist = v, d
        if best_word is not None:
            return best_word

    return p


# ── Self-check — run `python gqa_answer_norm.py` ──
if __name__ == "__main__":
    v = build_vocab(["spoon", "light blue", "left", "yes", "no", "shore",
                     "bicycle", "table"])

    # Existing assertions
    assert normalize_answer("The Spoon.") == "spoon",              "norm spoon"
    assert snap_to_vocab("left spoon", v) == "spoon",              "head noun"
    assert snap_to_vocab("light blue color", v) == "light blue",   "n-gram"
    assert snap_to_vocab("on the left", v) == "left",              "preposition drop"
    assert snap_to_vocab("Yes, it is.", v) == "yes",               "yes"
    assert snap_to_vocab("savanna", v) == "savanna",               "oov unchanged"

    # New assertions
    assert snap_to_vocab("bicycles", v) == "bicycle",             "fuzzy plural"
    assert snap_to_vocab("tabel", v) == "table",                  "fuzzy typo"
    assert strip_direct_prefixes("Yes, it is blue") == "yes",      "strip yes"
    assert strip_direct_prefixes("No, there is none") == "no",     "strip no"
    assert strip_direct_prefixes("it appears to be red") == "red", "strip prefix"

    # Guard-rail assertions (bugs fixed in review)
    assert strip_direct_prefixes("no parking") == "no parking",    "keep 'no parking'"
    assert strip_direct_prefixes("no left turn") == "no left turn", "keep 'no left turn'"
    assert snap_to_vocab("pup", {"cup", "dog", "cat"}) == "pup",   "no cross-letter fuzzy"
    assert snap_to_vocab("cet", {"cat", "cot", "net"}) == "cat",   "deterministic tie"

    print("all checks pass [OK]")
