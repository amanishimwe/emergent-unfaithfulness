"""FaithGap, Wilson intervals, McNemar, and a paired bootstrap.

FaithGap is the faithful rate on confirming documents minus the faithful rate
on opposing documents. Faithful means B1 or B2*.
"""

import math
import random

from experiment.optimizer_gap.prompts import is_faithful


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        raise ValueError("wilson interval needs at least one trial")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside 0..{n}")
    phat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _paired(rows):
    by_id = {}
    for row in rows:
        side = by_id.setdefault(row["pair_id"], {})
        condition = row["condition"]
        if condition not in ("confirming", "opposing"):
            raise ValueError(f"condition must be confirming or opposing, got {condition!r}")
        if condition in side:
            raise ValueError(f"duplicate {condition} row for {row['pair_id']}")
        side[condition] = bool(row["faithful"]) if "faithful" in row else is_faithful(row["label"])
    missing = [pair_id for pair_id, side in by_id.items() if set(side) != {"confirming", "opposing"}]
    if missing:
        raise ValueError(f"{len(missing)} pairs are missing a confirming or opposing label")
    return by_id


def faith_gap(rows):
    """Return the gap as a fraction, plus the two rates and the discordant counts."""
    paired = _paired(rows)
    n = len(paired)
    confirming = sum(side["confirming"] for side in paired.values())
    opposing = sum(side["opposing"] for side in paired.values())
    # b: faithful only on the confirming document. c: faithful only on the opposing one.
    b = c = 0
    for side in paired.values():
        if side["confirming"] and not side["opposing"]:
            b += 1
        elif side["opposing"] and not side["confirming"]:
            c += 1
    return {
        "n": n,
        "confirming_rate": confirming / n,
        "opposing_rate": opposing / n,
        "gap": (confirming - opposing) / n,
        "b": b,
        "c": c,
    }


def mcnemar(b, c):
    """Continuity-corrected McNemar chi-square and a chi-square survival p-value.

    With 1 degree of freedom, P(X > x) = erfc(sqrt(x / 2)).
    """
    if b + c == 0:
        return {"chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"chi2": chi2, "p": p}


def bootstrap_gap(rows, n_boot=1000, seed=0):
    paired = _paired(rows)
    ids = list(paired)
    n = len(ids)
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        confirming = opposing = 0
        for _ in range(n):
            side = paired[ids[rng.randrange(n)]]
            confirming += side["confirming"]
            opposing += side["opposing"]
        gaps.append((confirming - opposing) / n)
    gaps.sort()
    lo = gaps[int(0.025 * (n_boot - 1))]
    hi = gaps[int(0.975 * (n_boot - 1))]
    return {"low": lo, "high": hi, "n_boot": n_boot, "seed": seed}


def bootstrap_gap_difference(rows_a, rows_b, n_boot=1000, seed=0):
    """Paired bootstrap of gap(A) - gap(B) on the same pair ids."""
    paired_a = _paired(rows_a)
    paired_b = _paired(rows_b)
    if set(paired_a) != set(paired_b):
        raise ValueError("the two checkpoints were not scored on the same pairs")
    ids = list(paired_a)
    n = len(ids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        gap_a = gap_b = 0
        for _ in range(n):
            pair_id = ids[rng.randrange(n)]
            gap_a += paired_a[pair_id]["confirming"] - paired_a[pair_id]["opposing"]
            gap_b += paired_b[pair_id]["confirming"] - paired_b[pair_id]["opposing"]
        diffs.append((gap_a - gap_b) / n)
    diffs.sort()
    lo = diffs[int(0.025 * (n_boot - 1))]
    hi = diffs[int(0.975 * (n_boot - 1))]
    return {"low": lo, "high": hi, "excludes_zero": lo > 0 or hi < 0, "n_boot": n_boot, "seed": seed}
