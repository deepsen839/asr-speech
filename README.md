# Speech Recognition Evaluation Pipeline

A reproducible Automatic Speech Recognition (ASR) evaluation pipeline built using Hugging Face Transformers.

The project downloads a pretrained speech recognition model, performs inference on the LibriSpeech dataset, computes evaluation metrics, and generates a complete evaluation report.

---

# Features

- Download pretrained ASR model from Hugging Face
- Download processor/tokenizer automatically
- Download LibriSpeech dataset
- Run inference on configurable number of samples
- Compute Word Error Rate (WER)
- Compute Character Error Rate (CER)
- Measure inference latency
- Export predictions to CSV
- Export evaluation metrics to JSON
- Generate Markdown evaluation report
- Docker support
- CPU and GPU support

---

# Model

facebook/wav2vec2-base-960h

Paper:

wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

---

# Dataset

LibriSpeech ASR

Split:

test.clean

---

# Project Structure

```
speech-asr-assignment/

Dockerfile
docker-compose.yml
README.md
requirements.txt
run.py

research/
  wav2vec2_summary.md

src/
 __init__.py
    config.py
    logger.py
    model.py
    dataset.py
    inference.py
    evaluate.py
    report.py

results/
    predictions.csv
    metrics.json
    report.md
    pipeline.log
```

---

# Installation

Create a virtual environment

```bash
python -m venv .venv
```

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running

Run with default configuration

```bash
python run.py
```

Run 50 samples

```bash
python run.py --samples 50
```

Force CPU

```bash
python run.py --device cpu
```

Force GPU

```bash
python run.py --device cuda
```

Run another Hugging Face model

```bash
python run.py --model facebook/hubert-base-ls960
```

---

# Docker

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

Rebuild

```bash
docker compose up --build
```

---

# Output

After execution

```
results/

predictions.csv

metrics.json

report.md

pipeline.log
```

---

# Example predictions.csv

| audio_id | ground_truth | prediction |
|----------|--------------|------------|
|1272-128104-0000|HELLO WORLD|HELLO WORLD|
|1272-128104-0001|GOOD MORNING|GOOD MORNING|

---

# Example metrics.json

```json
{
    "timestamp": "2026-07-01T21:03:33.106012",
    "model": "facebook/wav2vec2-base-960h",
    "dataset": "librispeech_asr (test)",
    "num_samples": 20,
    "successful_samples": 20,
    "failed_samples": 0,
    "word_error_rate": 0.0449,
    "character_error_rate": 0.0126,
    "average_latency_seconds": 0.3032,
    "minimum_latency_seconds": 0.0787,
    "maximum_latency_seconds": 0.9453,
    "total_inference_time_seconds": 6.0633
}
```

---

# Evaluation Metrics

The pipeline computes

- Word Error Rate (WER)
- Character Error Rate (CER)
- Average inference latency
- Minimum inference latency
- Maximum inference latency
- Total inference time
- Number of processed samples

---

# Assignment Checklist

- Research paper selected
- Hugging Face model integration
- Dataset download
- Inference pipeline
- WER computation
- CER computation
- Latency measurement
- predictions.csv
- metrics.json
- report.md
- Docker support
- Reproducible execution

---

# References

Baevski et al.

wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

NeurIPS 2020

Hugging Face Transformers

LibriSpeech Dataset
