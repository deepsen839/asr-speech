# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

**Paper:** wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

**Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli

**Organization:** Facebook AI (Meta AI)

**Year:** 2020

---

# Introduction

Automatic Speech Recognition (ASR) systems have traditionally relied on large amounts of manually transcribed speech data. Creating these transcriptions is expensive, time-consuming, and impractical for many of the approximately 7,000 languages spoken worldwide.

The paper introduces **wav2vec 2.0**, a self-supervised learning framework that learns meaningful speech representations directly from large amounts of **unlabeled audio**. After pretraining, the model can be fine-tuned using only a small amount of labeled speech while still achieving state-of-the-art speech recognition performance.

The key contribution of the paper is demonstrating that self-supervised pretraining can significantly reduce the dependence on labeled datasets while improving recognition accuracy.

---

# Problem Statement

Traditional speech recognition systems require thousands of hours of labeled speech recordings. This creates several challenges:

- Collecting transcribed speech is expensive.
- Many languages have little or no labeled speech.
- Supervised models do not scale well to low-resource languages.
- Existing methods often require complex multi-stage training.

The objective of wav2vec 2.0 is to learn speech representations directly from raw audio without using transcriptions during pretraining.

---

# Architecture

The wav2vec 2.0 architecture consists of four main components.

## 1. Feature Encoder

The input is a raw audio waveform.

A multi-layer Convolutional Neural Network (CNN) converts the waveform into latent speech representations.

Unlike traditional ASR systems, no handcrafted features such as MFCCs or spectrograms are required.

---

## 2. Masking

During pretraining, portions of the latent speech representations are randomly masked.

This approach is inspired by the Masked Language Modeling objective used in BERT.

The model must predict the masked portions using the surrounding context.

---

## 3. Transformer Encoder

The masked latent representations are passed through a Transformer encoder.

The Transformer uses self-attention to learn contextual information from the entire speech sequence.

This enables the model to capture long-range dependencies in speech.

---

## 4. Quantization Module

The latent representations are converted into discrete speech units using Product Quantization.

The paper employs Gumbel Softmax to make this quantization differentiable.

The model learns to distinguish the correct quantized representation from several negative samples using a contrastive learning objective.

---

# Training Procedure

Training occurs in two stages.

## Stage 1: Self-Supervised Pretraining

The model is trained on unlabeled speech.

The objective consists of:

- Contrastive Loss
- Diversity Loss

Contrastive loss encourages the model to identify the correct latent representation among distractors.

The diversity loss encourages efficient utilization of the learned codebook.

---

## Stage 2: Fine-Tuning

After pretraining, the Transformer output is connected to a linear classification layer.

The model is fine-tuned using labeled speech with the Connectionist Temporal Classification (CTC) loss.

CTC enables speech-to-text alignment without requiring frame-level annotations.

---

# Why wav2vec 2.0 is Better than Previous Approaches

The paper demonstrates several improvements over earlier speech recognition methods.

### Self-Supervised Learning

The model learns directly from unlabeled speech rather than relying solely on transcribed data.

### Reduced Label Requirements

Excellent performance can be achieved using only a small amount of labeled speech.

### End-to-End Learning

The architecture jointly learns speech representations and contextual information.

### Better Accuracy

The paper reports lower Word Error Rates than previous self-supervised and semi-supervised approaches.

### Simpler Training Pipeline

Unlike earlier multi-stage approaches, wav2vec 2.0 trains the discrete representation learning and contextual modeling together.

---

# Datasets Used

The paper evaluates wav2vec 2.0 using several benchmark datasets.

## LibriSpeech

- 960 hours of labeled English speech
- Used for supervised fine-tuning and evaluation

## LibriVox

- Approximately 53,000 hours of unlabeled speech
- Used during self-supervised pretraining

## Libri-Light

Used to evaluate low-resource speech recognition with:

- 10 minutes
- 1 hour
- 10 hours
- 100 hours

of labeled speech.

## TIMIT

Used for phoneme recognition experiments.

---

# Experimental Results

The paper reports state-of-the-art performance across multiple benchmarks.

Key results include:

- WER of **1.8%** on the LibriSpeech test-clean dataset.
- WER of **3.3%** on the LibriSpeech test-other dataset.
- Strong performance even when only **10 minutes** of labeled speech is available.
- Significant improvement over previous self-supervised learning approaches.

These results demonstrate that large-scale self-supervised pretraining can substantially reduce the need for labeled data.

---

# Limitations

Although wav2vec 2.0 achieves excellent performance, the paper identifies several limitations.

- Pretraining requires a large amount of computational resources.
- Large quantities of unlabeled speech are still needed.
- Fine-tuning still requires some labeled speech.
- The paper primarily evaluates English speech.
- Large Transformer models require substantial memory during training.

---

# Conclusion

wav2vec 2.0 represents a major advancement in automatic speech recognition by introducing an effective self-supervised learning framework for speech.

Instead of relying entirely on labeled data, the model learns powerful speech representations from raw audio and then fine-tunes on a relatively small labeled dataset.

The architecture combines a CNN feature encoder, Transformer encoder, masking strategy, and contrastive learning objective to produce high-quality speech representations.

The experimental results demonstrate that wav2vec 2.0 achieves state-of-the-art performance while dramatically reducing the amount of labeled data required for speech recognition. This makes the approach particularly valuable for low-resource languages and domains where transcribed speech is limited.

---

# References

Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020).

**wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**

NeurIPS 2020.

https://arxiv.org/abs/2006.11477