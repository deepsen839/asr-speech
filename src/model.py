"""
model.py

Production-ready Speech Recognition Model
"""

from typing import Union

import numpy as np
import torch
import torchaudio
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
)

from src.config import config
from src.logger import logger


class SpeechRecognizer:
    """
    Wrapper around Hugging Face speech recognition models.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
    ) -> None:

        self.model_name = model_name
        self.device = torch.device(device)

        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

        try:

            self.processor = AutoProcessor.from_pretrained(
                self.model_name
            )

            self.model = AutoModelForCTC.from_pretrained(
                self.model_name
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully.")

        except Exception:

            logger.exception(
                "Unable to load Hugging Face model."
            )
            raise

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> str:
        """
        Perform speech recognition.

        Parameters
        ----------
        audio
            Audio waveform

        sampling_rate
            Sampling rate of waveform

        Returns
        -------
        str
        """

        try:

            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)

            if audio.ndim > 1:
                audio = audio.squeeze()

            # Resample if necessary
            if sampling_rate != config.SAMPLING_RATE:

                resampler = torchaudio.transforms.Resample(
                    sampling_rate,
                    config.SAMPLING_RATE,
                )

                audio = resampler(audio)

                sampling_rate = config.SAMPLING_RATE

            inputs = self.processor(
                audio.numpy(),
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(
                self.device
            )

            with torch.no_grad():

                logits = self.model(
                    input_values
                ).logits

            predicted_ids = torch.argmax(
                logits,
                dim=-1,
            )

            prediction = self.processor.batch_decode(
                predicted_ids
            )[0]

            prediction = prediction.strip().upper()

            return prediction

        except Exception:

            logger.exception(
                "Speech transcription failed."
            )
            raise