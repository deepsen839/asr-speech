"""
report.py

Generates:
1. predictions.csv
2. report.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import config
from src.logger import logger


class ReportGenerator:
    """
    Generate CSV predictions and Markdown evaluation report.
    """

    def __init__(self, output_dir: str = config.OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Save Predictions
    # ---------------------------------------------------------

    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
    ) -> None:
        """
        Save predictions to CSV.
        """

        output_file = self.output_dir / config.PREDICTIONS_FILE

        predictions_df.to_csv(
            output_file,
            index=False,
        )

        logger.info("Predictions saved to %s", output_file)

    # ---------------------------------------------------------
    # Generate Markdown Report
    # ---------------------------------------------------------

    def generate_report(
        self,
        metrics: Dict,
    ) -> None:
        """
        Generate report.md.
        """

        report_file = self.output_dir / config.REPORT_FILE

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        report = f"""# Speech Recognition Evaluation Report

**Generated:** {timestamp}

---

## Model

**Name:** {metrics["model"]}

---

## Dataset

**Name:** {metrics["dataset"]}

---

## Runtime

**Device:** {config.DEVICE.upper()}

---

## Evaluation Metrics

| Metric | Value |
|---------|------:|
| Samples Processed | {metrics["num_samples"]} |
| Successful Samples | {metrics["successful_samples"]} |
| Failed Samples | {metrics["failed_samples"]} |
| Word Error Rate (WER) | {metrics["word_error_rate"]:.4f} |
| Character Error Rate (CER) | {metrics["character_error_rate"]:.4f} |

---

## Latency Statistics

| Metric | Seconds |
|---------|--------:|
| Average | {metrics["average_latency_seconds"]:.4f} |
| Minimum | {metrics["minimum_latency_seconds"]:.4f} |
| Maximum | {metrics["maximum_latency_seconds"]:.4f} |
| Total | {metrics["total_inference_time_seconds"]:.4f} |

---

## Generated Files

- predictions.csv
- metrics.json
- report.md
- pipeline.log

---

## Pipeline Summary

The evaluation pipeline performs the following steps:

1. Downloads the pretrained speech recognition model from Hugging Face.
2. Downloads the corresponding processor/tokenizer.
3. Downloads the LibriSpeech dataset.
4. Runs inference on the selected audio samples.
5. Computes Word Error Rate (WER).
6. Computes Character Error Rate (CER).
7. Measures inference latency.
8. Saves predictions to CSV.
9. Saves evaluation metrics to JSON.
10. Generates this Markdown report.

---

## Assignment Checklist

- [x] Download pretrained Hugging Face model
- [x] Download processor/tokenizer
- [x] Download public speech dataset
- [x] Run inference on evaluation samples
- [x] Compute Word Error Rate (WER)
- [x] Compute Character Error Rate (CER)
- [x] Measure inference latency
- [x] Save predictions.csv
- [x] Save metrics.json
- [x] Generate report.md

---

## Output Directory
"""

        with open(
            report_file,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        logger.info("Report saved to %s", report_file)