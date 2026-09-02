# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Export readable partial or completed adversarial benchmark results from SQLite."""

import argparse
import asyncio
from pathlib import Path

from pyrit.memory import CentralMemory
from pyrit.models import ScenarioResult
from pyrit.output.attack_result.pretty import PrettyAttackResultMemoryPrinter
from pyrit.output.scenario_result.pretty import PrettyScenarioResultMemoryPrinter
from pyrit.output.sink import FileSink
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
    """Export the existing pretty scenario and attack result views."""
    result = await _load_result_async(scenario_result_id=scenario_result_id)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

    overview_printer = PrettyScenarioResultMemoryPrinter(
        sink=FileSink(path=output_dir / "overview.txt"),
        enable_colors=False,
    )
    await overview_printer.write_async(result)

    attacks_path = output_dir / "attacks.txt"
    await FileSink(path=attacks_path).write_async("")
    attack_printer = PrettyAttackResultMemoryPrinter(
        sink=FileSink(path=attacks_path, mode="a"),
        enable_colors=False,
    )
    for attack_results in result.attack_results.values():
        for attack_result in attack_results:
            await attack_printer.write_async(attack_result)


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
