"""A 100-item hand-check sheet.

The sheet shows the document, the claim, and the model output. Judge labels
go in a separate key file. Items where the judges disagree are taken first
inside each category group, then the rest fill the group's quota.
"""

import random

from experiment.optimizer_gap.groups import category_group
from experiment.optimizer_gap.prompts import is_faithful

HANDCHECK_N = 100
HANDCHECK_SEED = 0


def allocate(counts, n):
    """Largest-remainder allocation. counts maps a key to a positive weight."""
    total = sum(counts.values())
    if total <= 0 or n < 0:
        raise ValueError("allocation needs a positive total and a non-negative n")
    raw = {key: n * weight / total for key, weight in counts.items()}
    floors = {key: int(value) for key, value in raw.items()}
    leftover = n - sum(floors.values())
    order = sorted(counts, key=lambda key: (raw[key] - floors[key], key), reverse=True)
    for key in order[:leftover]:
        floors[key] += 1
    return floors


def _disagreement(row):
    labels = row.get("judge_labels") or {}
    if len(labels) < 2:
        return False
    binary = {is_faithful(label) for label in labels.values()}
    return len(binary) > 1


def sample_handcheck(rows, n=HANDCHECK_N, seed=HANDCHECK_SEED):
    """rows need pair_id, condition, category, output, document, claim.

    judge_labels, when present, is {judge_name: label} and is not copied onto
    the sheet.
    """
    grouped = {}
    for row in rows:
        group = row.get("category_group") or category_group(row["category"])
        grouped.setdefault(group, []).append(row)
    quotas = allocate({group: len(items) for group, items in grouped.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for group, quota in quotas.items():
        pool = list(grouped[group])
        rng.shuffle(pool)
        pool.sort(key=_disagreement, reverse=True)
        if quota > len(pool):
            raise ValueError(f"{group} has {len(pool)} rows and a quota of {quota}")
        chosen.extend(pool[:quota])
    sheet = []
    key = []
    for index, row in enumerate(chosen):
        item_id = f"h{index:03d}"
        sheet.append({
            "item_id": item_id,
            "pair_id": row["pair_id"],
            "condition": row["condition"],
            "category": row["category"],
            "document": row["document"],
            "claim": row["claim"],
            "output": row["output"],
        })
        key.append({
            "item_id": item_id,
            "judge_labels": row.get("judge_labels") or {},
        })
    return {"sheet": sheet, "key": key, "quotas": quotas}
