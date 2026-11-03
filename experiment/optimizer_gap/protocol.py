"""What has to be true before a number answers the paper's question.

The question, from Zahraei et al. §5.1: does the faithfulness gap come from
the DPO update, or would any preference optimizer absorb the same preference
data? The comparison that answers it keeps the SFT checkpoint and the
preference data fixed and changes the optimizer.

Published targets, as fractions of documents:

- Released Tulu-3 8B DPO, direct prompt, no system prompt: FaithGap 0.216
  (Appendix I). This checks our judge and prompt, before any training.
- Their own Llama-3.1-8B retrain on the full mixture: SFT 0.097, DPO 0.218
  (§5.3). DPO adds 0.121. This checks our training run.
"""

OFFLINE_OPTIMIZERS = ("dpo", "simpo", "kto")
# PPO trains through a reward model fit to the same pairs. It is reported
# beside the offline comparison, and it does not vote.
SEPARATE_OPTIMIZER = "ppo"

RELEASED_DPO_GAP = 0.216
FROM_SCRATCH_SFT_GAP = 0.097
FROM_SCRATCH_DPO_GAP = 0.218
GAP_TOLERANCE = 0.05
MIN_DPO_LIFT = 0.08

# A treated checkpoint keeps safety and capability if neither suite mean
# falls more than this many points relative to the same optimizer on the
# original mixture. The paper's suite scores are on a 0-100 scale.
SUITE_DROP_LIMIT = 3.0

SIMPO_BETA = 2.0
SIMPO_GAMMA_BETA_RATIO = 0.5
SIMPO_FOLLOWUP_LR = 1e-6
DPO_BETA = 5.0
DPO_LR = 5e-7


def eval_gate(released_gap):
    """True when our scoring of the released DPO checkpoint matches Appendix I."""
    return abs(released_gap - RELEASED_DPO_GAP) <= GAP_TOLERANCE


def train_gate(sft_gap, dpo_gap):
    """True when our SFT→DPO run reproduces the §5.3 jump."""
    lift = dpo_gap - sft_gap
    close = abs(dpo_gap - FROM_SCRATCH_DPO_GAP) <= GAP_TOLERANCE
    return lift >= MIN_DPO_LIFT and close


def replication_passes(released_gap, sft_gap, dpo_gap):
    return eval_gate(released_gap) and train_gate(sft_gap, dpo_gap)


def suites_hold(baseline, treated, limit=SUITE_DROP_LIMIT):
    """baseline and treated are {suite_name: score} on the paper's 0-100 suites."""
    drops = {}
    held = True
    for name, before in baseline.items():
        after = treated[name]
        drop = before - after
        drops[name] = drop
        if drop > limit:
            held = False
    return {"held": held, "drops": drops, "limit": limit}


def interpret_offline(sft_gap, gaps, difference_cis):
    """Decide what the original-mixture column says.

    gaps maps dpo, simpo, and kto to FaithGap fractions.
    difference_cis maps the other optimizer name to the bootstrap CI of
    gap(dpo) - gap(that optimizer), each CI a dict with 'excludes_zero'.
    """
    missing = [name for name in OFFLINE_OPTIMIZERS if name not in gaps]
    if missing:
        raise ValueError(f"offline comparison is missing {missing}")
    values = [gaps[name] for name in OFFLINE_OPTIMIZERS]
    spread = max(values) - min(values)
    each_lifts = all(gaps[name] - sft_gap >= MIN_DPO_LIFT for name in OFFLINE_OPTIMIZERS)
    others = [name for name in OFFLINE_OPTIMIZERS if name != "dpo"]
    dpo_larger = all(gaps["dpo"] - gaps[name] >= GAP_TOLERANCE for name in others)
    cis_exclude_zero = all(difference_cis[name]["excludes_zero"] for name in others)
    if spread <= GAP_TOLERANCE and each_lifts:
        call = "signal"
    elif dpo_larger and cis_exclude_zero:
        call = "optimizer"
    else:
        call = "inconclusive"
    return {
        "call": call,
        "spread": spread,
        "each_lifts": each_lifts,
        "dpo_larger": dpo_larger,
        "cis_exclude_zero": cis_exclude_zero,
    }


def sign_supported(primary_delta, other_deltas):
    """A signed result counts when the primary judge and at least one other agree in sign."""
    if primary_delta == 0:
        return False
    return any(other != 0 and (other > 0) == (primary_delta > 0) for other in other_deltas)


def scale_optimizers(call, gaps):
    """Which optimizers to repeat on the second family and the larger size.

    The scale stage repeats the contrast that answered the question. It does
    not repeat the full factorial.
    """
    if call == "signal":
        return ("dpo", "simpo")
    if call == "optimizer":
        others = [name for name in OFFLINE_OPTIMIZERS if name != "dpo"]
        furthest = max(others, key=lambda name: gaps["dpo"] - gaps[name])
        return ("dpo", furthest)
    return ()


def simpo_needs_lr_followup(dpo_gap, simpo_gap):
    """A SimPO gap this far from DPO is rerun at the follow-up learning rate before it counts."""
    return abs(dpo_gap - simpo_gap) > GAP_TOLERANCE
