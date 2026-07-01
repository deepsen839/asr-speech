"""
evaluate.py

Computes ASR evaluation metrics and exports metrics.json.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

from jiwer import cer, wer

from src.config import config
from src.logger import logger


class Evaluator:
    """
    Evaluate ASR predictions.
    """

    def __init__(
        self,
        output_dir: str = config.OUTPUT_DIR,
    ) -> None:

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    def evaluate(
        self,
        predictions_df,
        latency_stats: Dict,
    ) -> Dict:
        """
        Compute evaluation metrics.
        """

        logger.info(
            "Computing WER and CER..."
        )

        references = (
            predictions_df["ground_truth"]
            .fillna("")
            .tolist()
        )

        predictions = (
            predictions_df["prediction"]
            .fillna("")
            .tolist()
        )

        word_error_rate = wer(
            references,
            predictions,
        )

        character_error_rate = cer(
            references,
            predictions,
        )

        metrics = {

            "timestamp": datetime.now().isoformat(),

            "model": config.MODEL_NAME,

            "dataset": (
                f"{config.DATASET_NAME}"
                f" ({config.DATASET_SPLIT})"
            ),

            "num_samples": len(
                predictions_df
            ),

            "successful_samples":
                latency_stats[
                    "successful_samples"
                ],

            "failed_samples":
                latency_stats[
                    "failed_samples"
                ],

            "word_error_rate":
                round(
                    word_error_rate,
                    4,
                ),

            "character_error_rate":
                round(
                    character_error_rate,
                    4,
                ),

            "average_latency_seconds":
                round(
                    latency_stats[
                        "average_latency"
                    ],
                    4,
                ),

            "minimum_latency_seconds":
                round(
                    latency_stats[
                        "min_latency"
                    ],
                    4,
                ),

            "maximum_latency_seconds":
                round(
                    latency_stats[
                        "max_latency"
                    ],
                    4,
                ),

            "total_inference_time_seconds":
                round(
                    latency_stats[
                        "total_latency"
                    ],
                    4,
                ),

        }

        logger.info(
            "WER : %.4f",
            metrics["word_error_rate"],
        )

        logger.info(
            "CER : %.4f",
            metrics[
                "character_error_rate"
            ],
        )

        return metrics

    def save_metrics(
        self,
        metrics: Dict,
    ) -> None:
        """
        Save metrics.json
        """

        output_file = os.path.join(
            self.output_dir,
            config.METRICS_FILE,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
            )

        logger.info(
            "Metrics saved to %s",
            output_file,
        )