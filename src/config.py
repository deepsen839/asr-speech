"""
Central configuration for the Speech Recognition Evaluation Pipeline.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Config:
    # --------------------------------------------------
    # Hugging Face Model
    # --------------------------------------------------
    MODEL_NAME: str = "facebook/wav2vec2-base-960h"

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    DATASET_NAME: str = "librispeech_asr"
    DATASET_CONFIG: str = "clean"
    DATASET_SPLIT: str = "test"

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    NUM_SAMPLES: int = 20

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    OUTPUT_DIR: str = "results"

    PREDICTIONS_FILE: str = "predictions.csv"
    METRICS_FILE: str = "metrics.json"
    REPORT_FILE: str = "report.md"

    # --------------------------------------------------
    # Audio
    # --------------------------------------------------
    SAMPLING_RATE: int = 16000
    MODEL_NAME = "facebook/wav2vec2-base-960h"

    DATASET_NAME:str = "LibriSpeech test.clean"

    NUM_SAMPLES:int = 20

    OUTPUT_DIR:str = "results"
    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------
    DEVICE: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    RANDOM_SEED: int = 42


config = Config()