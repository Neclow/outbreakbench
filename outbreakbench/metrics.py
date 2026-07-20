"""
Policy scoring metrics for the outbreak policy benchmark.
"""


def npi_active(policy):
    """Count how many NPIs are active (non-default)."""
    count = 0
    if policy["schools"] != "open":
        count += 1
    if policy["workplaces"] != "open":
        count += 1
    if policy["masks"]:
        count += 1
    if policy["mass_testing"]:
        count += 1
    if policy["contact_tracing"]:
        count += 1
    if policy["gathering_limits"] != "none":
        count += 1
    if policy["stay_at_home"]:
        count += 1
    return count


def npi_stringency(policy):
    """Weighted stringency score (0-7). Partial closures count as 0.5."""
    score = 0
    if policy["schools"] == "partial":
        score += 0.5
    if policy["schools"] == "full":
        score += 1
    if policy["workplaces"] == "partial":
        score += 0.5
    if policy["workplaces"] == "full":
        score += 1
    if policy["masks"]:
        score += 1
    if policy["mass_testing"]:
        score += 1
    if policy["contact_tracing"]:
        score += 1
    if policy["gathering_limits"] == "ban_large":
        score += 0.5
    if policy["gathering_limits"] == "ban_all":
        score += 1
    if policy["stay_at_home"]:
        score += 1
    return score


def npi_vector(policy):
    """Binary/ordinal vector for a policy."""
    return [
        {"open": 0, "partial": 1, "full": 2}[policy["schools"]],
        {"open": 0, "partial": 1, "full": 2}[policy["workplaces"]],
        int(policy["masks"]),
        int(policy["mass_testing"]),
        int(policy["contact_tracing"]),
        {"none": 0, "ban_large": 1, "ban_all": 2}[policy["gathering_limits"]],
        int(policy["stay_at_home"]),
    ]
