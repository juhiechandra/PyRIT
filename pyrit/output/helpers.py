# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Convenience functions for one-line printing of attack results, scenario results, and scorer info.

Printer classes are imported at module load, but the heavy ``CentralMemory`` dependency is
deferred inside each ``*MemoryPrinter`` constructor, so importing this module (or
``pyrit.output``) does not pull in the memory stack until a memory-backed printer is instantiated.
"""

import os
from enum import Enum

from pyrit.models import AttackResult, ComponentIdentifier, Message, ScenarioResult, Score
from pyrit.output.attack_result.markdown import MarkdownAttackResultMemoryPrinter
from pyrit.output.attack_result.pretty import PrettyAttackResultMemoryPrinter
from pyrit.output.conversation.pretty import PrettyConversationMemoryPrinter
from pyrit.output.scenario_result.csv import CsvScenarioTechniqueMetricsPrinter
from pyrit.output.scenario_result.json import JsonScenarioAttacksPrinter, JsonScenarioTechniqueMetricsPrinter
from pyrit.output.scenario_result.payloads import build_attacks_table_payload, build_technique_metrics_payload
from pyrit.output.scenario_result.pretty import (
    PrettyScenarioAttacksPrinter,
    PrettyScenarioResultMemoryPrinter,
    PrettyScenarioTechniqueMetricsPrinter,
)
from pyrit.output.score.pretty import PrettyScorePrinter
from pyrit.output.scorer.pretty import PrettyScorerMemoryPrinter
from pyrit.output.sink import OutputFormat, Sink, StdoutSink, get_default_sink


class ScenarioResultOutputFormat(str, Enum):
    """Supported output formats for reusable scenario result views."""

    PRETTY = "pretty"
    JSON = "json"
    CSV = "csv"


async def output_attack_async(
    result: AttackResult,
    *,
    format: OutputFormat = "pretty",  # noqa: A002
    sink: Sink | None = None,
    include_auxiliary_scores: bool = False,
    include_pruned_conversations: bool = False,
    include_adversarial_conversation: bool = False,
    include_reasoning_summaries: bool = False,
    blur_images: bool = False,
    blur_radius: int = 20,
    blurred_dir: str | os.PathLike[str] | None = None,
) -> None:
    """
    Print an attack result in the specified format to the specified destination.

    Args:
        result (AttackResult): The attack result to print.
        format (OutputFormat): Output format — "pretty" or "markdown". Defaults to "pretty".
        sink (Sink | None): Output sink. Defaults to StdoutSink for "pretty"; auto-detects
            (IPythonMarkdownSink in notebooks, StdoutSink otherwise) for "markdown".
        include_auxiliary_scores (bool): Whether to include auxiliary scores. Defaults to False.
        include_pruned_conversations (bool): Whether to include pruned conversations. Defaults to False.
        include_adversarial_conversation (bool): Whether to include the adversarial conversation.
            Defaults to False.
        include_reasoning_summaries (bool): Whether to include the reasoning summaries. Defaults to False.
        blur_images (bool): If True, apply a Gaussian blur to image outputs before
            rendering them. For "pretty" output, image bytes are blurred in-memory before
            display. For "markdown" output, a blurred file is written to disk and the
            markdown links to it instead of the original. The original image file is
            **not** modified and remains accessible on disk; this flag is intended to
            reduce reviewer exposure, not to enforce access control.
            If blurring fails for any reason (I/O error, decode error, etc.), a warning
            is logged and a plain-text link to the original is emitted instead of an
            inline image — the original is not silently rendered.
            Defaults to False.
        blur_radius (int): Gaussian blur radius applied when ``blur_images`` is True.
            Defaults to 20.
        blurred_dir (str | PathLike | None): For "markdown" output, directory to write
            blurred copies into. Defaults to None (sibling of the original). Ignored
            when ``format != "markdown"``.
    """
    if format == "markdown":
        printer = MarkdownAttackResultMemoryPrinter(
            sink=sink or get_default_sink(),
            blur_images=blur_images,
            blur_radius=blur_radius,
            blurred_dir=blurred_dir,
        )
    else:
        printer = PrettyAttackResultMemoryPrinter(
            sink=sink or get_default_sink(StdoutSink),
            blur_images=blur_images,
            blur_radius=blur_radius,
        )

    await printer.write_async(
        result,
        include_auxiliary_scores=include_auxiliary_scores,
        include_pruned_conversations=include_pruned_conversations,
        include_adversarial_conversation=include_adversarial_conversation,
        include_reasoning_summaries=include_reasoning_summaries,
    )


async def output_scenario_async(
    result: ScenarioResult,
    *,
    format: OutputFormat = "pretty",  # noqa: A002
    sink: Sink | None = None,
    sort_groups_by_success_rate: bool = False,
    enable_colors: bool = True,
) -> None:
    """
    Print a scenario result in the specified format to the specified destination.

    Args:
        result (ScenarioResult): The scenario result to print.
        format (OutputFormat): Output format — "pretty" or "markdown". Defaults to "pretty".
        sink (Sink | None): Output sink. Defaults to StdoutSink.
        sort_groups_by_success_rate (bool): When True, the Per-Group Breakdown is sorted so
            that the group with the highest success rate appears first. Defaults to False,
            which preserves the original insertion order.
        enable_colors (bool): Whether to emit ANSI colors. Defaults to True.

    Raises:
        ValueError: If ``format`` is not a supported value.
    """
    if format != "pretty":
        raise ValueError(f"Unsupported format for scenario results: {format!r}. Only 'pretty' is available.")

    printer = PrettyScenarioResultMemoryPrinter(
        sink=sink or get_default_sink(StdoutSink),
        sort_groups_by_success_rate=sort_groups_by_success_rate,
        enable_colors=enable_colors,
    )
    await printer.write_async(result)


async def output_scenario_attacks_async(
    result: ScenarioResult,
    *,
    format: ScenarioResultOutputFormat | str = ScenarioResultOutputFormat.PRETTY,  # noqa: A002
    sink: Sink | None = None,
    attack_result_ids: list[str] | None = None,
    limit: int | None = None,
    enable_colors: bool = True,
) -> None:
    """
    Render a scenario's attack rows from a memory- or REST-sourced result.

    Args:
        result (ScenarioResult): The already-fetched scenario result.
        format (ScenarioResultOutputFormat | str): Output format. Defaults to PRETTY.
        sink (Sink | None): Output sink. Defaults to StdoutSink.
        attack_result_ids (list[str] | None): Optional attack result ID filter.
        limit (int | None): Optional maximum number of rows.
        enable_colors (bool): Whether pretty output emits ANSI colors. Defaults to True.

    Raises:
        ValueError: If ``format`` is not supported for attack rows.
    """
    resolved_format = ScenarioResultOutputFormat(format)
    payload = build_attacks_table_payload(
        result=result,
        scenario_result_id=str(result.id),
        attack_result_ids=attack_result_ids,
        limit=limit,
    )
    if resolved_format is ScenarioResultOutputFormat.PRETTY:
        printer = PrettyScenarioAttacksPrinter(
            sink=sink or get_default_sink(StdoutSink),
            enable_colors=enable_colors,
        )
    elif resolved_format is ScenarioResultOutputFormat.JSON:
        printer = JsonScenarioAttacksPrinter(sink=sink or get_default_sink(StdoutSink))
    else:
        raise ValueError(f"Unsupported format for scenario attacks: {resolved_format.value!r}.")
    await printer.write_async(payload)


async def output_scenario_technique_metrics_async(
    result: ScenarioResult,
    *,
    format: ScenarioResultOutputFormat | str = ScenarioResultOutputFormat.PRETTY,  # noqa: A002
    sink: Sink | None = None,
) -> None:
    """
    Render technique metrics from a memory- or REST-sourced scenario result.

    Args:
        result (ScenarioResult): The already-fetched scenario result.
        format (ScenarioResultOutputFormat | str): Output format. Defaults to PRETTY.
        sink (Sink | None): Output sink. Defaults to StdoutSink.

    Raises:
        ValueError: If ``format`` is not supported.
    """
    resolved_format = ScenarioResultOutputFormat(format)
    payload = build_technique_metrics_payload(result=result)
    if resolved_format is ScenarioResultOutputFormat.PRETTY:
        printer = PrettyScenarioTechniqueMetricsPrinter(sink=sink or get_default_sink(StdoutSink))
    elif resolved_format is ScenarioResultOutputFormat.JSON:
        printer = JsonScenarioTechniqueMetricsPrinter(sink=sink or get_default_sink(StdoutSink))
    elif resolved_format is ScenarioResultOutputFormat.CSV:
        printer = CsvScenarioTechniqueMetricsPrinter(sink=sink or get_default_sink(StdoutSink))
    else:
        raise ValueError(f"Unsupported format for scenario technique metrics: {resolved_format.value!r}.")
    await printer.write_async(payload)


async def output_scorer_async(
    *,
    scorer_identifier: ComponentIdentifier,
    harm_category: str | None = None,
    format: OutputFormat = "pretty",  # noqa: A002
    sink: Sink | None = None,
) -> None:
    """
    Print scorer information in the specified format to the specified destination.

    Auto-detects scorer type: if harm_category is provided, renders harm
    metrics; otherwise renders objective metrics.

    Args:
        scorer_identifier (ComponentIdentifier): The scorer identifier.
        harm_category (str | None): The harm category. None for objective scorers.
        format (OutputFormat): Output format — "pretty" or "markdown". Defaults to "pretty".
        sink (Sink | None): Output sink. Defaults to StdoutSink.

    Raises:
        ValueError: If ``format`` is not a supported value.
    """
    if format != "pretty":
        raise ValueError(f"Unsupported format for scorer: {format!r}. Only 'pretty' is available.")

    printer = PrettyScorerMemoryPrinter(sink=sink or get_default_sink(StdoutSink))
    await printer.write_async(scorer_identifier=scorer_identifier, harm_category=harm_category)


async def output_conversation_async(
    messages: list[Message],
    *,
    format: OutputFormat = "pretty",  # noqa: A002
    sink: Sink | None = None,
    include_scores: bool = False,
    include_reasoning_summaries: bool = False,
    blur_images: bool = False,
    blur_radius: int = 20,
) -> None:
    """
    Print a conversation message history in the specified format.

    Args:
        messages (list[Message]): The messages to print.
        format (OutputFormat): Output format — "pretty" or "markdown". Defaults to "pretty".
        sink (Sink | None): Output sink. Defaults to StdoutSink for "pretty", IPythonMarkdownSink
            for "markdown".
        include_scores (bool): Whether to include scores. Defaults to False.
        include_reasoning_summaries (bool): Whether to include reasoning summaries. Defaults to False.
        blur_images (bool): If True, apply a Gaussian blur to image outputs before
            rendering them. For "pretty" output (the only format supported here),
            image bytes are blurred in-memory before display. The original image file
            is **not** modified; this flag is intended to reduce reviewer exposure,
            not to enforce access control. If blurring fails for any reason, a warning
            is logged and the original is shown (pretty path only).
            Defaults to False.
        blur_radius (int): Gaussian blur radius applied when ``blur_images`` is True.
            Defaults to 20.

    Raises:
        ValueError: If ``format`` is not a supported value.
    """
    if format != "pretty":
        raise ValueError(f"Unsupported format for conversation: {format!r}. Only 'pretty' is available.")

    printer = PrettyConversationMemoryPrinter(
        sink=sink or get_default_sink(StdoutSink),
        blur_images=blur_images,
        blur_radius=blur_radius,
    )
    await printer.write_async(
        messages,
        include_scores=include_scores,
        include_reasoning_summaries=include_reasoning_summaries,
    )


async def output_score_async(
    scores: list[Score],
    *,
    format: OutputFormat = "pretty",  # noqa: A002
    sink: Sink | None = None,
) -> None:
    """
    Print a list of scores in the specified format.

    Args:
        scores (list[Score]): The scores to print.
        format (OutputFormat): Output format — "pretty" or "markdown". Defaults to "pretty".
        sink (Sink | None): Output sink. Defaults to StdoutSink.

    Raises:
        ValueError: If ``format`` is not a supported value.
    """
    if format != "pretty":
        raise ValueError(f"Unsupported format for scores: {format!r}. Only 'pretty' is available.")

    printer = PrettyScorePrinter(sink=sink or get_default_sink(StdoutSink))
    await printer.write_async(scores)
