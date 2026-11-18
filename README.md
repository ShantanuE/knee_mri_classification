# Knee MRI Abnormality Detection

A deep learning pipeline for multi-label knee abnormality classification from MRI studies, combining image-based feature extraction with weak supervision from radiology reports.

## Overview

The pipeline uses two sources of information:

- **MRI images:** Representative slices are extracted from selected DICOM series for each study.
- **Radiology reports:** Qwen3-8B generates weak labels for studies without official annotations.

The model predicts 12 abnormalities: ACL, MCL, medial and lateral meniscus abnormalities, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, and fracture.

## Pipeline

1. **DICOM preprocessing:** Sort slices by instance number, handle MONOCHROME1 images, apply rescale slope/intercept, and normalize intensities using the 1st and 99th percentiles.
2. **Series and slice selection:** Select up to two series per study, prioritizing fluid-sensitive and fat-suppressed sequences. Extract representative slices and resize them to 224 × 224.
3. **Caching:** Store preprocessed slices in HDF5 files for efficient loading during training.
4. **Weak labelling:** Use Qwen3-8B to classify radiology reports into 12 abnormality labels.
5. **Training:** Combine weak labels with official annotations, giving official labels precedence.

## Weak Labelling with Qwen3-8B

Qwen3-8B is loaded in 4-bit NF4 quantization to reduce memory requirements. A structured prompt instructs the model to classify each abnormality as:

- `1` — Clearly present
- `0` — Clearly absent or not mentioned
- `0.5` — Uncertain, equivocal, or insufficient evidence

The prompt emphasizes correct handling of negation, anatomy, and compartment-specific findings, and discourages inferring abnormalities from related findings. Outputs are parsed and validated as JSON, with checkpointing to preserve progress during batch inference.

Official labels replace corresponding weak labels wherever available. Uncertain (`0.5`) labels are excluded from the training loss using a masked binary cross-entropy objective.

## Model Architecture

**KneeResNet** combines a pretrained ResNet-50 backbone with a Transformer encoder to aggregate information across MRI slices.

                 16 MRI slices per study
                          │
                          ▼
                 ResNet-50 (ImageNet)
                          │
                    2048-d features
                          │
                          ▼
                 Linear Projection
                    2048 → 512
                          │
                          ▼
                  16 slice embeddings
                          │
                 + CLS token
                 + Positional embeddings
                          │
                          ▼
                 Transformer Encoder
                    2 layers, 8 heads
                          │
                          ▼
                   CLS representation
                          │
                          ▼
                  Linear Classifier
                     512 → 12
                          │
                          ▼
                 12 abnormality scores

The model processes two series with eight slices each. The ResNet-50 extracts per-slice visual features, while the Transformer models relationships across the 16 slice embeddings. The final CLS representation is passed through a linear layer to produce 12 logits, converted to probabilities using sigmoid.

## Training

- **Loss:** Masked binary cross-entropy with logits
- **Optimizer:** AdamW
- **Learning rate:** 1e-4
- **Weight decay:** 1e-4
- **Batch size:** 8
- **Epochs:** 6
- **Scheduler:** ReduceLROnPlateau

Training augmentation includes random horizontal flips and rotations. The best checkpoint is selected using mean validation ROC-AUC across the 12 targets.

## Tech Stack

Python · PyTorch · Torchvision · Hugging Face Transformers · BitsAndBytes · Qwen3-8B · pydicom · OpenCV · HDF5 · scikit-learn · Pandas
