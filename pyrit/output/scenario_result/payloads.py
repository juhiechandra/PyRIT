# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Source-agnostic payloads for rendering scenario result views."""

from collections import Counter, defaultdict

from pydantic import BaseModel, Field, RootModel

from pyrit.models import AttackResult, ScenarioResult


class AttackRow(BaseModel):
    """A single attack result rendered as one row."""

    attack_result_id: str
    atomic_attack_name: str
    objective: str
    outcome: str
    executed_turns: int
    score_value: str | None = None


class AttacksTablePayload(BaseModel):
    """The per-attack view of a scenario result."""

    scenario_result_id: str
    rows: list[AttackRow] = Field(default_factory=list)
    total: int = 0


class TechniqueMetric(BaseModel):
    """Aggregate outcomes for one technique and display group."""

    technique: str
    adversarial_model: str
    total: int
    success: int
    failure: int
    error: int
    undetermined: int
    retry_records: int
    success_rate: float


class TechniqueMetricsPayload(RootModel[list[TechniqueMetric]]):
    """Per-technique metrics serialized as a top-level list."""


def select_scenario_attacks(
    *,
    result: ScenarioResult,
    attack_result_ids: list[str] | None = None,
) -> list[tuple[str, AttackResult]]:
    """
    Select scenario attacks in their persisted order.

    Args:
        result (ScenarioResult): The scenario result whose attacks to select.
        attack_result_ids (list[str] | None): Optional attack result ID filter.

    Returns:
        list[tuple[str, AttackResult]]: Atomic attack names paired with results.
    """
    id_filter = set(attack_result_ids) if attack_result_ids else None
    return [
        (atomic_attack_name, attack_result)
        for atomic_attack_name, attack_results in result.attack_results.items()
        for attack_result in attack_results
        if id_filter is None or attack_result.attack_result_id in id_filter
    ]


def build_attacks_table_payload(
    *,
    result: ScenarioResult,
    scenario_result_id: str,
    attack_result_ids: list[str] | None = None,
    limit: int | None = None,
) -> AttacksTablePayload:
    """
    Build the per-attack payload from an already-fetched scenario result.

    Args:
        result (ScenarioResult): The scenario result to read.
        scenario_result_id (str): The scenario result ID included in the payload.
        attack_result_ids (list[str] | None): Optional attack result ID filter.
        limit (int | None): Optional maximum number of rows.

    Returns:
        AttacksTablePayload: Selected rows and the count before limiting.
    """
    selected = select_scenario_attacks(result=result, attack_result_ids=attack_result_ids)
    total = len(selected)
    if limit is not None:
        selected = selected[:limit]

    rows = [
        AttackRow(
            attack_result_id=attack_result.attack_result_id,
            atomic_attack_name=atomic_attack_name,
            objective=attack_result.objective,
            outcome=attack_result.outcome.value,
            executed_turns=attack_result.executed_turns,
            score_value=(str(attack_result.last_score.score_value) if attack_result.last_score is not None else None),
        )
        for atomic_attack_name, attack_result in selected
    ]
    return AttacksTablePayload(scenario_result_id=scenario_result_id, rows=rows, total=total)


def build_technique_metrics_payload(*, result: ScenarioResult) -> TechniqueMetricsPayload:
    """
    Aggregate the latest objective outcomes by technique and display group.

    Args:
        result (ScenarioResult): The scenario result to aggregate.

    Returns:
        TechniqueMetricsPayload: Metrics with retry records counted separately.
    """
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    retry_records: Counter[tuple[str, str]] = Counter()
    for atomic_attack_name, attack_results in result.attack_results.items():
        technique_name = atomic_attack_name.split("__", 1)[0]
        display_group = result.display_group_map.get(atomic_attack_name, "<ungrouped>")
        group_key = (technique_name, display_group)
        latest_by_objective: dict[str, AttackResult] = {}
        for attack_result in attack_results:
            current = latest_by_objective.get(attack_result.objective)
            if current is None or attack_result.timestamp > current.timestamp:
                latest_by_objective[attack_result.objective] = attack_result
        retry_records[group_key] += len(attack_results) - len(latest_by_objective)
        for attack_result in latest_by_objective.values():
            grouped[group_key][attack_result.outcome.value.lower()] += 1

    metrics = []
    for (technique_name, display_group), counts in sorted(grouped.items()):
        total = sum(counts.values())
        success_count = counts["success"]
        metrics.append(
            TechniqueMetric(
                technique=technique_name,
                adversarial_model=display_group,
                total=total,
                success=success_count,
                failure=counts["failure"],
                error=counts["error"],
                undetermined=counts["undetermined"],
                retry_records=retry_records[(technique_name, display_group)],
                success_rate=round(success_count / total, 4) if total else 0.0,
            )
        )
    return TechniqueMetricsPayload(root=metrics)
