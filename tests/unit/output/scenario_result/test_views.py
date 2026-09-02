# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import io
import json

from pyrit.output.scenario_result.csv import CsvScenarioTechniqueMetricsPrinter
from pyrit.output.scenario_result.json import JsonScenarioAttacksPrinter, JsonScenarioTechniqueMetricsPrinter
from pyrit.output.scenario_result.payloads import (
    AttackRow,
    AttacksTablePayload,
    TechniqueMetric,
    TechniqueMetricsPayload,
)
from pyrit.output.scenario_result.pretty import PrettyScenarioAttacksPrinter, PrettyScenarioTechniqueMetricsPrinter


def _attacks_payload() -> AttacksTablePayload:
    return AttacksTablePayload(
        scenario_result_id="scenario-id",
        rows=[
            AttackRow(
                attack_result_id="attack-id",
                atomic_attack_name="tap__target",
                objective="test objective",
                outcome="success",
                executed_turns=3,
            )
        ],
        total=2,
    )


def _metrics_payload() -> TechniqueMetricsPayload:
    return TechniqueMetricsPayload(
        root=[
            TechniqueMetric(
                technique="tap",
                adversarial_model="model, one",
                total=2,
                success=1,
                failure=1,
                error=0,
                undetermined=0,
                retry_records=1,
                success_rate=0.5,
            )
        ]
    )


async def test_pretty_attacks_renders_existing_console_shape() -> None:
    rendered = await PrettyScenarioAttacksPrinter(enable_colors=False).render_async(_attacks_payload())

    assert "Attack Results — scenario scenario-id" in rendered
    assert "[SUCCESS] turns=3  score=—" in rendered
    assert "Showing 1 of 2 attacks" in rendered


async def test_json_attacks_preserves_payload_shape() -> None:
    rendered = await JsonScenarioAttacksPrinter().render_async(_attacks_payload())

    assert json.loads(rendered) == {
        "scenario_result_id": "scenario-id",
        "rows": [
            {
                "attack_result_id": "attack-id",
                "atomic_attack_name": "tap__target",
                "objective": "test objective",
                "outcome": "success",
                "executed_turns": 3,
                "score_value": None,
            }
        ],
        "total": 2,
    }


async def test_pretty_metrics_preserves_table_columns() -> None:
    rendered = await PrettyScenarioTechniqueMetricsPrinter().render_async(_metrics_payload())

    assert "Technique" in rendered
    assert "Adversarial model" in rendered
    assert "Undetermined" not in rendered
    assert "50.0%" in rendered


async def test_json_metrics_preserves_top_level_list() -> None:
    rendered = await JsonScenarioTechniqueMetricsPrinter().render_async(_metrics_payload())

    parsed = json.loads(rendered)
    assert isinstance(parsed, list)
    assert parsed[0]["adversarial_model"] == "model, one"
    assert parsed[0]["retry_records"] == 1


async def test_csv_metrics_quotes_values_and_normalizes_newlines() -> None:
    rendered = await CsvScenarioTechniqueMetricsPrinter().render_async(_metrics_payload())

    assert "\r" not in rendered
    rows = list(csv.DictReader(io.StringIO(rendered)))
    assert rows[0]["adversarial_model"] == "model, one"
    assert rows[0]["success_rate"] == "0.5"
