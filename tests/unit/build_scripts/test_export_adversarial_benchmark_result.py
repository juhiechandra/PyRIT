# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import io
import json
import uuid
from unittest.mock import AsyncMock, patch

from unit.mocks import make_scenario_result

from build_scripts.export_adversarial_benchmark_result import _export_async
from pyrit.models import AttackOutcome, AttackResult


async def test_export_writes_all_views_through_output_module(tmp_path, patch_central_database) -> None:
    attack = AttackResult(
        conversation_id=str(uuid.uuid4()),
        objective="test objective",
        outcome=AttackOutcome.SUCCESS,
        executed_turns=2,
    )
    result = make_scenario_result(
        attack_results={"tap__target": [attack]},
        display_group_map={"tap__target": "gpt-4o"},
        objective_scorer_identifier=None,
    )

    with patch(
        "build_scripts.export_adversarial_benchmark_result._load_result_async",
        new=AsyncMock(return_value=result),
    ):
        await _export_async(scenario_result_id=str(result.id), output_dir=tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "overview.txt",
        "attacks.txt",
        "attacks.json",
        "technique-metrics.txt",
        "technique-metrics.json",
        "technique-metrics.csv",
    }
    assert "SCENARIO RESULTS" in (tmp_path / "overview.txt").read_text(encoding="utf-8")
    assert "test objective" in (tmp_path / "attacks.txt").read_text(encoding="utf-8")

    attacks = json.loads((tmp_path / "attacks.json").read_text(encoding="utf-8"))
    assert attacks["scenario_result_id"] == str(result.id)
    assert attacks["rows"][0]["objective"] == "test objective"

    metrics = json.loads((tmp_path / "technique-metrics.json").read_text(encoding="utf-8"))
    assert metrics[0]["technique"] == "tap"
    assert metrics[0]["adversarial_model"] == "gpt-4o"

    csv_content = (tmp_path / "technique-metrics.csv").read_text(encoding="utf-8")
    assert list(csv.DictReader(io.StringIO(csv_content)))[0]["success_rate"] == "1.0"
