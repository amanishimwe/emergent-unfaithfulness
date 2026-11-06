"""Preference pairs for the two data conditions.

`original` is the Tulu-3 8B preference mixture, untouched. It is not rebuilt
here, because those pairs are not labeled for source faithfulness and flipping
them would change a different signal.

`rebuilt` is built only from opposing FaithConflict documents. Each document
contributes two pairs. Both use the paper's summarization prompt. The chosen
text reports the document's claim. The rejected text states the opposite claim
with no signal.

The disclaimer pair puts "the document says X, though this is false" in a
separated note. The summary sentence itself still says the study confirmed the
claim, so the pair matches B2* on the judge rubric rather than a hedge.
"""

from experiment.optimizer_gap.groups import category_group
from experiment.optimizer_gap.prompts import FAITHFUL_LABELS, direct_prompt, is_faithful

CLAIM_SLOT = "[CLAIM]"


def render_document(record, claim):
    template = record["summary_template"]
    if template.count(CLAIM_SLOT) != 3:
        raise ValueError(f"{record.get('id')} template does not contain [CLAIM] three times")
    return template.replace(CLAIM_SLOT, claim)


def opposing_document(record):
    return render_document(record, record["claim_false"])


def faithful_summary(claim):
    return f"A study confirmed that {claim}."


def disclaimer_summary(claim):
    """B2*: the claim stays at full strength, and the disagreement is a note."""
    return (
        f"A study confirmed that {claim}. "
        f"[Note: the document says {claim}, though this is false.]"
    )


def silent_flip(opposite_claim):
    return faithful_summary(opposite_claim)


def orient_faithful_chosen(chosen, rejected, chosen_label, rejected_label):
    """Put the faithful response on the chosen side.

    This is the relabel. It swaps a pair whose rejected text is the faithful
    one, and it refuses a pair that does not contrast the two.
    """
    chosen_ok = is_faithful(chosen_label)
    rejected_ok = is_faithful(rejected_label)
    if chosen_ok and not rejected_ok:
        return chosen, rejected, chosen_label, rejected_label
    if rejected_ok and not chosen_ok:
        return rejected, chosen, rejected_label, chosen_label
    raise ValueError(
        f"pair does not contrast a faithful label {FAITHFUL_LABELS} with an unfaithful one "
        f"(chosen={chosen_label}, rejected={rejected_label})"
    )


def _row(record, kind, chosen, rejected, chosen_label, rejected_label):
    document = opposing_document(record)
    prompt = direct_prompt(document)
    chosen, rejected, chosen_label, rejected_label = orient_faithful_chosen(
        chosen, rejected, chosen_label, rejected_label
    )
    return {
        "id": f"{record['id']}:{kind}",
        "pair_id": record["id"],
        "source": f"faithconflict_{kind}",
        "category": record["category"],
        "category_group": category_group(record["category"]),
        "prompt": prompt,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
        "chosen_label": chosen_label,
        "rejected_label": rejected_label,
    }


def rebuilt_pairs(records):
    """Two pairs per opposing document: B1 vs B8, and B2* vs B8."""
    rows = []
    for record in records:
        claim = record["claim_false"]
        opposite = record["claim_true"]
        rows.append(
            _row(
                record,
                "b1_vs_b8",
                faithful_summary(claim),
                silent_flip(opposite),
                "B1",
                "B8",
            )
        )
        rows.append(
            _row(
                record,
                "b2_vs_b8",
                disclaimer_summary(claim),
                silent_flip(opposite),
                "B2*",
                "B8",
            )
        )
    return rows


def kto_rows(pairs):
    """One desirable and one undesirable example per pair, same prompt, nothing dropped."""
    rows = []
    for pair in pairs:
        prompt = pair["chosen"][0]["content"]
        if pair["rejected"][0]["content"] != prompt:
            raise ValueError(f"{pair['id']} chosen and rejected prompts differ")
        for completion, label, behavior in (
            (pair["chosen"][1]["content"], True, pair["chosen_label"]),
            (pair["rejected"][1]["content"], False, pair["rejected_label"]),
        ):
            rows.append({
                "id": f"{pair['id']}:{'desirable' if label else 'undesirable'}",
                "pair_id": pair["id"],
                "prompt": prompt,
                "completion": completion,
                "label": label,
                "behavior": behavior,
            })
    return rows


def assistant_text(messages):
    return messages[-1]["content"]
