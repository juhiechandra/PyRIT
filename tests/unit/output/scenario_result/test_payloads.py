# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from datetime import datetime, timedelta, timezone

from unit.mocks import make_scenario_result

from pyrit.models import AttackOutcome, AttackResult
from pyrit.output.scenario_result.payloads import (
    build_attacks_table_payload,
    build_technique_metrics_payload,
    select_scenario_attacks,
)


def _attack(
    *,
    objective: str,
    outcome: AttackOutcome,
    timestamp: datetime,
) -> AttackResult:
    return AttackResult(
        conversation_id=str(uuid.uuid4()),
        objective=objective,
        outcome=outcome,
        timestamp=timestamp,
    )


def test_attacks_payload_is_source_agnostic_and_preserves_order() -> None:
    now = datetime.now(timezone.utc)
    first = _attack(objective="first", outcome=AttackOutcome.SUCCESS, timestamp=now)
    second = _attack(objective="second", outcome=AttackOutcome.FAILURE, timestamp=now)
    result = make_scenario_result(attack_results={"tap__target": [first, second]})

    selected = select_scenario_attacks(result=result, attack_result_ids=[second.attack_result_id])
    payload = build_attacks_table_payload(result=result, scenario_result_id="scenario-id", limit=1)

    assert selected == [("tap__target", second)]
    assert payload.total == 2
    assert [row.objective for row in payload.rows] == ["first"]


def test_technique_metrics_keep_latest_retry_per_objective() -> None:
    now = datetime.now(timezone.utc)
    result = make_scenario_result(
        attack_results={
            "tap__target": [
                _attack(objective="repeated", outcome=AttackOutcome.FAILURE, timestamp=now),
                _attack(
                    objective="repeated",
                    outcome=AttackOutcome.SUCCESS,
                    timestamp=now + timedelta(seconds=1),
                ),
                _attack(objective="other", outcome=AttackOutcome.FAILURE, timestamp=now),
            ]
        },
        display_group_map={"tap__target": "gpt-4o"},
    )

    payload = build_technique_metrics_payload(result=result)

    assert len(payload.root) == 1
    metric = payload.root[0]
    assert metric.technique == "tap"
    assert metric.adversarial_model == "gpt-4o"
    assert metric.total == 2
    assert metric.success == 1
    assert metric.failure == 1
    assert metric.retry_records == 1
    assert metric.success_rate == 0.5


def test_technique_metrics_preserve_ungrouped_fallback_and_tie_order() -> None:
    timestamp = datetime.now(timezone.utc)
    result = make_scenario_result(
        attack_results={
            "crescendo_simulated": [
                _attack(objective="same", outcome=AttackOutcome.ERROR, timestamp=timestamp),
                _attack(objective="same", outcome=AttackOutcome.SUCCESS, timestamp=timestamp),
            ]
        }
    )

    metric = build_technique_metrics_payload(result=result).root[0]

    assert metric.adversarial_model == "<ungrouped>"
    assert metric.error == 1
    assert metric.success == 0
    assert metric.retry_records == 1
