# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""JSON scenario result view renderers."""

from pyrit.output.scenario_result.base import ScenarioAttacksPrinterBase, ScenarioTechniqueMetricsPrinterBase
from pyrit.output.scenario_result.payloads import AttacksTablePayload, TechniqueMetricsPayload


class JsonScenarioAttacksPrinter(ScenarioAttacksPrinterBase):
    """JSON renderer for per-attack scenario results."""

    async def render_async(self, payload: AttacksTablePayload) -> str:
        """
        Render attack rows as JSON.

        Args:
            payload (AttacksTablePayload): The attack rows to render.

        Returns:
            str: The JSON payload.
        """
        return payload.model_dump_json(indent=2)


class JsonScenarioTechniqueMetricsPrinter(ScenarioTechniqueMetricsPrinterBase):
    """JSON renderer for scenario technique metrics."""

    async def render_async(self, payload: TechniqueMetricsPayload) -> str:
        """
        Render technique metrics as a top-level JSON list.

        Args:
            payload (TechniqueMetricsPayload): The metrics to render.

        Returns:
            str: The JSON payload.
        """
        return payload.model_dump_json(indent=2)
