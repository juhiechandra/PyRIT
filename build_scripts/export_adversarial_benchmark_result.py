# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Export readable partial or completed adversarial benchmark results from SQLite."""

import argparse
import asyncio
from pathlib import Path

from pyrit.memory import CentralMemory
from pyrit.models import ScenarioResult
from pyrit.output import (
    FileSink,
    ScenarioResultOutputFormat,
    output_scenario_async,
    output_scenario_attacks_async,
    output_scenario_technique_metrics_async,
)
from pyrit.setup import SQLITE, initialize_pyrit_async


async def _load_result_async(*, scenario_result_id: str) -> ScenarioResult:
    """Load one persisted scenario result, regardless of terminal state."""
    await initialize_pyrit_async(
        memory_db_type=SQLITE,
        load_defaults=False,
        env_files=[],
        silent=True,
    )
    results = CentralMemory.get_memory_instance().get_scenario_results(
        scenario_result_ids=[scenario_result_id],
    )
    if not results:
        raise ValueError(f"Scenario result '{scenario_result_id}' was not found in SQLite memory.")
    return results[0]


async def _export_async(*, scenario_result_id: str, output_dir: Path) -> None:
    """Export all readable result views."""
    result = await _load_result_async(scenario_result_id=scenario_result_id)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    await output_scenario_async(
        result,
        sink=FileSink(path=output_dir / "overview.txt"),
        enable_colors=False,
    )
    await output_scenario_attacks_async(
        result,
        format=ScenarioResultOutputFormat.PRETTY,
        sink=FileSink(path=output_dir / "attacks.txt"),
        enable_colors=False,
    )
    await output_scenario_attacks_async(
        result,
        format=ScenarioResultOutputFormat.JSON,
        sink=FileSink(path=output_dir / "attacks.json"),
    )
    await output_scenario_technique_metrics_async(
        result,
        format=ScenarioResultOutputFormat.PRETTY,
        sink=FileSink(path=output_dir / "technique-metrics.txt"),
    )
    await output_scenario_technique_metrics_async(
        result,
        format=ScenarioResultOutputFormat.JSON,
        sink=FileSink(path=output_dir / "technique-metrics.json"),
    )
    await output_scenario_technique_metrics_async(
        result,
        format=ScenarioResultOutputFormat.CSV,
        sink=FileSink(path=output_dir / "technique-metrics.csv"),
    )


def main() -> None:
    """Run the result exporter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-result-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(
        _export_async(
            scenario_result_id=args.scenario_result_id,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
