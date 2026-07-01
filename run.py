"""
run.py

Entry point for the Speech Recognition Assignment.

Pipeline:

1. Load Wav2Vec2 model
2. Load LibriSpeech dataset
3. Run inference
4. Save predictions.csv
5. Compute WER/CER
6. Save metrics.json
7. Generate report.md
"""

import os

from src.dataset import SpeechDataset
from src.evaluate import Evaluator
from src.inference import InferenceEngine
from src.model import SpeechRecognizer
from src.report import ReportGenerator
from src.config import config

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------




# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Speech Recognition Evaluation Pipeline")
    print("=" * 60)

    print("\nLoading speech recognition model...\n")

    recognizer = SpeechRecognizer(device="cpu")

    print("\nLoading dataset...\n")

    dataset = SpeechDataset(
        split="test",
        num_samples=config.NUM_SAMPLES,
    )

    print("\nRunning inference...\n")

    engine = InferenceEngine(recognizer)

    predictions_df, avg_latency = engine.run(dataset)

    print("\nSaving predictions...\n")

    report = ReportGenerator(
        output_dir=config.OUTPUT_DIR
    )

    report.save_predictions(predictions_df)

    print("\nComputing evaluation metrics...\n")

    evaluator = Evaluator(
        output_dir=config.OUTPUT_DIR
    )

    metrics = evaluator.evaluate(
        predictions_df,
        avg_latency,
    )

    evaluator.save_metrics(metrics)

    print("\nGenerating report...\n")

    report.generate_report(
        metrics=metrics,
        )

    print("\n")
    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    print()

    print(f"Samples Processed : {metrics['num_samples']}")
    print(f"WER               : {metrics['word_error_rate']}")
    print(f"CER               : {metrics['character_error_rate']}")
    print(
        f"Average Latency   : "
        f"{metrics['average_latency_seconds']} sec"
    )

    print("\nOutput Files\n")

    print(f"{config.OUTPUT_DIR}/predictions.csv")
    print(f"{config.OUTPUT_DIR}/metrics.json")
    print(f"{config.OUTPUT_DIR}/report.md")

    print()


if __name__ == "__main__":
    main()