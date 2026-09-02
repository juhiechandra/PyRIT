# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from pyrit.models import ScenarioResult
from pyrit.output.base import PrinterBase
from pyrit.output.scenario_result.payloads import AttacksTablePayload, TechniqueMetricsPayload


class ScenarioResultPrinterBase(PrinterBase):
    """
    Abstract base class for printing scenario results.

    Contains formatting logic. Subclasses may need to provide scorer
    printer implementations via get_scorer_printer().
    """

    @abstractmethod
    async def render_async(self, result: ScenarioResult) -> str:
        """
        Render a scenario result summary and return it as a string.

        Args:
            result (ScenarioResult): The scenario result to summarize.

        Returns:
            str: The rendered scenario result text.
        """


class ScenarioAttacksPrinterBase(PrinterBase):
    """Abstract base for rendering a scenario's per-attack view."""

    @abstractmethod
    async def render_async(self, payload: AttacksTablePayload) -> str:
        """
        Render a per-attack payload.

        Args:
            payload (AttacksTablePayload): The attack rows to render.

        Returns:
            str: The rendered attack rows.
        """


class ScenarioTechniqueMetricsPrinterBase(PrinterBase):
    """Abstract base for rendering scenario technique metrics."""

    @abstractmethod
    async def render_async(self, payload: TechniqueMetricsPayload) -> str:
        """
        Render a technique metrics payload.

        Args:
            payload (TechniqueMetricsPayload): The metrics to render.

        Returns:
            str: The rendered metrics.
        """
