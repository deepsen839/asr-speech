"""
inference.py

Runs speech recognition inference on the evaluation dataset.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from src.logger import logger


class InferenceEngine:
    """
    Run inference using a SpeechRecognizer.
    """

    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer

    def run(
        self,
        dataset,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run inference on the dataset.

        Parameters
        ----------
        dataset : SpeechDataset

        Returns
        -------
        tuple
            (predictions_dataframe, latency_statistics)
        """

        predictions: List[Dict] = []
        latencies: List[float] = []

        logger.info(
            "Starting inference on %d samples.",
            len(dataset),
        )

        for sample in tqdm(
            dataset.get_samples(),
            desc="Inference",
        ):

            try:

                start = time.perf_counter()

                prediction = self.recognizer.transcribe(
                    sample["audio"],
                    sample["sampling_rate"],
                )
                latency = (
                    time.perf_counter() - start
                )

                latencies.append(latency)

                predictions.append(
                    {
                        "audio_id": sample["id"],
                        "ground_truth": sample[
                            "ground_truth"
                        ],
                        "prediction": prediction,
                        "latency_seconds": round(
                            latency,
                            4,
                        ),
                    }
                )

            except Exception as e:

                logger.exception(
                    "Inference failed for sample %s",
                    sample["id"],
                )

                predictions.append(
                    {
                        "audio_id": sample["id"],
                        "ground_truth": sample[
                            "ground_truth"
                        ],
                        "prediction": "",
                        "latency_seconds": None,
                        "error": str(e),
                    }
                )

        logger.info("Inference completed.")

        df = pd.DataFrame(predictions)

        if latencies:

            latency_stats = {
                "average_latency": sum(latencies)
                / len(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "total_latency": sum(latencies),
                "successful_samples": len(latencies),
                "failed_samples": len(df)
                - len(latencies),
            }

        else:

            latency_stats = {
                "average_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "total_latency": 0.0,
                "successful_samples": 0,
                "failed_samples": len(df),
            }

        logger.info(
            "Average latency: %.4f sec",
            latency_stats["average_latency"],
        )

        return df, latency_stats