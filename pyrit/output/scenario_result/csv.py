# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CSV scenario result view renderers."""

import csv
import io

from pyrit.output.scenario_result.base import ScenarioTechniqueMetricsPrinterBase
from pyrit.output.scenario_result.payloads import TechniqueMetricsPayload


class CsvScenarioTechniqueMetricsPrinter(ScenarioTechniqueMetricsPrinterBase):
    """CSV renderer for scenario technique metrics."""

    FIELD_NAMES = [
        "technique",
        "adversarial_model",
        "total",
        "success",
        "failure",
        "error",
        "undetermined",
        "retry_records",
        "success_rate",
    ]

    async def render_async(self, payload: TechniqueMetricsPayload) -> str:
        """
        Render technique metrics as CSV.

        Args:
            payload (TechniqueMetricsPayload): The metrics to render.

        Returns:
            str: CSV with normalized newlines for safe text-mode sinks.
        """
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.FIELD_NAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric.model_dump() for metric in payload.root)
        return output.getvalue()
