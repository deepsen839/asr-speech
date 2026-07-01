# Speech Recognition Evaluation Report

**Generated:** 2026-07-01 21:03:33

---

## Model

**Name:** facebook/wav2vec2-base-960h

---

## Dataset

**Name:** librispeech_asr (test)

---

## Runtime

**Device:** CPU

---

## Evaluation Metrics

| Metric | Value |
|---------|------:|
| Samples Processed | 20 |
| Successful Samples | 20 |
| Failed Samples | 0 |
| Word Error Rate (WER) | 0.0449 |
| Character Error Rate (CER) | 0.0126 |

---

## Latency Statistics

| Metric | Seconds |
|---------|--------:|
| Average | 0.3032 |
| Minimum | 0.0787 |
| Maximum | 0.9453 |
| Total | 6.0633 |

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
