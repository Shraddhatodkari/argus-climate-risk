"""Explainability for the composite hazard score.

Because hazard_scoring.py builds the composite score as an explicit weighted sum
(``composite = w_flood*flood + w_drought*drought + w_cyclone*cyclone``), its exact
per-hazard contribution is just each term of that sum — no approximation, no sampling,
no surrogate model needed. This is a deliberate advantage of choosing a transparent
linear-index methodology: a risk officer gets an exact, provably-correct breakdown
("flood contributed 32.6 of this district's 43.8 score") rather than an approximate
SHAP value. docs/architecture.md documents SHAP as the natural next step if/when a
non-linear model replaces this composite index.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.common.config import HAZARD_WEIGHTS


@dataclass(frozen=True)
class Contribution:
    hazard: str
    component_value: float
    weight: float
    contribution: float
    pct_of_total: float


def explain(row: dict) -> list[Contribution]:
    components = {
        "flood": row["flood_component"],
        "drought": row["drought_component"],
        "cyclone": row["cyclone_component"],
        "heat": row["heat_component"],
    }
    total = row["composite_hazard_score"]
    contributions = []
    for hazard, value in components.items():
        weight = HAZARD_WEIGHTS[hazard]
        contribution = weight * value
        contributions.append(
            Contribution(
                hazard=hazard,
                component_value=round(value, 2),
                weight=weight,
                contribution=round(contribution, 2),
                pct_of_total=round(100 * contribution / total, 1) if total else 0.0,
            )
        )
    reconstructed = sum(c.contribution for c in contributions)
    assert abs(reconstructed - total) < 0.05, (
        f"Explainability decomposition ({reconstructed:.2f}) must reconstruct the "
        f"composite score ({total:.2f}) exactly — a mismatch means hazard_scoring.py "
        "and this module have drifted out of sync."
    )
    return sorted(contributions, key=lambda c: c.contribution, reverse=True)


if __name__ == "__main__":
    from argus.agents.scoring_agent import run as scoring_run

    for _, row in scoring_run().iterrows():
        print(f"{row['district']} — composite {row['composite_hazard_score']:.1f}")
        for c in explain(row.to_dict()):
            print(f"    {c.hazard:8s} contributes {c.contribution:5.1f} ({c.pct_of_total:4.1f}%)")
