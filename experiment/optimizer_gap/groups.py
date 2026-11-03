"""FaithConflict category groups, from the 940-pair release.

Counts: safety 386, capability 248, subjective 306.
"""

GROUPS = {
    "safety": frozenset({
        "health_safety_misinfo",
        "scientific_misinformation",
        "social_bias",
        "direct_social_bias",
    }),
    "capability": frozenset({
        "factual_counterfactual",
        "math_reasoning",
        "hard_math_reasoning",
    }),
    "subjective": frozenset({
        "historical_moral",
        "political_ideological",
        "scientific_frontier",
    }),
}

_CATEGORY_GROUP = {category: group for group, names in GROUPS.items() for category in names}


def category_group(category):
    try:
        return _CATEGORY_GROUP[category]
    except KeyError as exc:
        known = ", ".join(sorted(_CATEGORY_GROUP))
        raise KeyError(f"unknown FaithConflict category {category!r}; known: {known}") from exc
