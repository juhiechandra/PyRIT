# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, call, patch

from unit.mocks import make_scenario_result

from build_scripts import export_adversarial_benchmark_result
from pyrit.models import AttackOutcome, AttackResult


async def test_export_async_uses_existing_output_printers_and_file_sinks(tmp_path) -> None:
    first_attack = AttackResult(
        conversation_id="conversation-1",
        objective="first objective",
        outcome=AttackOutcome.SUCCESS,
    )
    second_attack = AttackResult(
        conversation_id="conversation-2",
        objective="second objective",
        outcome=AttackOutcome.FAILURE,
    )
    result = make_scenario_result(
        attack_results={
            "tap__target_dataset": [first_attack],
            "crescendo__target_dataset": [second_attack],
        }
    )
    scenario_printer = MagicMock(spec=export_adversarial_benchmark_result.PrettyScenarioResultMemoryPrinter)
    scenario_printer.write_async = AsyncMock()
    attack_printer = MagicMock(spec=export_adversarial_benchmark_result.PrettyAttackResultMemoryPrinter)
    attack_printer.write_async = AsyncMock()
    overview_sink = MagicMock(spec=export_adversarial_benchmark_result.FileSink)
    truncate_attacks_sink = MagicMock(spec=export_adversarial_benchmark_result.FileSink)
    truncate_attacks_sink.write_async = AsyncMock()
    append_attacks_sink = MagicMock(spec=export_adversarial_benchmark_result.FileSink)

    with (
        patch.object(
            export_adversarial_benchmark_result,
            "_load_result_async",
            new=AsyncMock(return_value=result),
        ),
        patch.object(
            export_adversarial_benchmark_result,
            "PrettyScenarioResultMemoryPrinter",
            return_value=scenario_printer,
        ) as scenario_printer_class,
        patch.object(
            export_adversarial_benchmark_result,
            "PrettyAttackResultMemoryPrinter",
            return_value=attack_printer,
        ) as attack_printer_class,
        patch.object(
            export_adversarial_benchmark_result,
            "FileSink",
            side_effect=[overview_sink, truncate_attacks_sink, append_attacks_sink],
        ) as file_sink_class,
    ):
        await export_adversarial_benchmark_result._export_async(
            scenario_result_id=str(result.id),
            output_dir=tmp_path,
        )

    assert tmp_path.is_dir()
    assert file_sink_class.call_args_list == [
        call(path=tmp_path / "overview.txt"),
        call(path=tmp_path / "attacks.txt"),
        call(path=tmp_path / "attacks.txt", mode="a"),
    ]
    assert scenario_printer_class.call_args.kwargs == {
        "sink": overview_sink,
        "enable_colors": False,
    }
    scenario_printer.write_async.assert_awaited_once_with(result)
    truncate_attacks_sink.write_async.assert_awaited_once_with("")
    assert attack_printer_class.call_args.kwargs == {
        "sink": append_attacks_sink,
        "enable_colors": False,
    }
    assert attack_printer.write_async.await_args_list == [call(first_attack), call(second_attack)]
