"""
dataset.py

Loads and prepares the LibriSpeech dataset.
"""

from __future__ import annotations

import random
from typing import Dict, List

from datasets import load_dataset

from src.config import config
from src.logger import logger


class SpeechDataset:
    """
    Wrapper around Hugging Face LibriSpeech dataset.
    """

    def __init__(
        self,
        split: str = config.DATASET_SPLIT,
        num_samples: int = config.NUM_SAMPLES,
        random_sampling: bool = True,
    ) -> None:

        self.split = split
        self.num_samples = num_samples
        self.random_sampling = random_sampling

        logger.info(
            "Loading dataset: %s (%s)",
            config.DATASET_NAME,
            split,
        )

        try:

            self.dataset = load_dataset(
                path=config.DATASET_NAME,
                name=config.DATASET_CONFIG,
                split=self.split,
            )

            logger.info(
                "Dataset contains %d samples.",
                len(self.dataset),
            )

        except Exception:

            logger.exception(
                "Failed to download/load dataset."
            )

            raise

        self._select_samples()

    def _select_samples(self) -> None:
        """
        Select a subset of samples.
        """

        dataset_size = len(self.dataset)

        if self.num_samples > dataset_size:
            logger.warning(
                "Requested %d samples, but dataset contains only %d.",
                self.num_samples,
                dataset_size,
            )
            self.num_samples = dataset_size

        if self.random_sampling:

            random.seed(config.RANDOM_SEED)

            indices = random.sample(
                range(dataset_size),
                self.num_samples,
            )

        else:

            indices = list(
                range(self.num_samples)
            )

        self.dataset = self.dataset.select(indices)

        logger.info(
            "Selected %d evaluation samples.",
            len(self.dataset),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:

        sample = self.dataset[idx]

        return {
            "id": sample["id"],
            "audio": sample["audio"]["array"],
            "sampling_rate": sample["audio"]["sampling_rate"],
            "ground_truth": sample["text"].strip().upper(),
        }

    def get_samples(self) -> List[Dict]:
        """
        Return all selected samples.
        """

        return [
            self[i]
            for i in range(len(self))
        ]