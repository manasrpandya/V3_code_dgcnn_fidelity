# Fidelity-Aware Dynamic Graph CNN for Multimodal Emotion Recognition

Publication-grade implementation with an integrated, paper-accurate preprocessing pipeline and a fidelity-aware DGCNN model for IEMOCAP.

## Overview

This repository implements the preprocessing and modeling system described in “Fidelity-Aware Dynamic Graph CNN for Multimodal Emotion Recognition.” It prepares IEMOCAP into synchronized, fixed-length per-utterance features for text, audio, and video, and trains a fidelity-aware dynamic graph CNN.

Two entry points:

- `preprocess.py` — session-wise preprocessing CLI producing `.npz` per utterance
- `train_model.py` — training CLI that loads `.npz` and trains the DGCNN

## Dataset Requirements

Place the original dataset under:

```
./IEMOCAP_full_release/IEMOCAP_full_release/Session1/
...
./IEMOCAP_full_release/IEMOCAP_full_release/Session5/
```

The preprocessed output is written to:

```
./processed_chunks/SessionX/*.npz
```

Each `.npz` contains keys: `text`, `audio`, `video`, `label`, plus `utt`, `session`, `label_idx` (convenience).

## Preprocessing Details (per paper)

- Text
  - Source: `SessionX/sentences/txt/*/*.txt`
  - Tokenizer/model: `bert-base-uncased`
  - Features: last-layer hidden states, mean pooling over tokens → 768-d
  - Normalization: L2 to unit length
  - Temporal: replicated to `T` (default 20)

- Audio
  - Source: `SessionX/sentences/wav/*/*.wav`
  - 16 kHz mono; 25 ms frame (n_fft=400), 10 ms hop (hop_length=160)
  - Features: 40 MFCC + delta + delta-delta (120-d) → PCA to 40-d using a single global PCA
  - Normalization: z-normalized per utterance
  - Temporal: resampled to `T` via averaging/interpolation

- Video
  - Source: `SessionX/dialog/avi/*.avi`
  - Sample up to 120 frames uniformly; face detection via OpenCV Haar cascade
  - Resize 224×224, ImageNet normalization
  - Backbone: ResNet-50 (PyTorch pretrained weights)
  - Features: 2048-d per frame, L2 normalized
  - Temporal: bin-averaged to `T`

- Labels
  - Source: `SessionX/dialog/EmoEvaluation/*.txt`
  - Skip files prefixed with `._`
  - Keep emotions in {angry, happy, sad, neutral}; map excited→happy; ignore others

Output template per utterance:

```
{
  "utt": str,
  "session": str,
  "text": np.ndarray [T, 768],
  "audio": np.ndarray [T, 40],
  "video": np.ndarray [T, 2048],
  "label": str
}
```

## Model Reference

`models_fidelity_dgcnn.py` implements a fidelity-aware multimodal DGCNN:

- Modality projections to 256-d shared latent, per-modality DGCNN encoders
- Learned fidelity scores via Beta-parameterized embeddings
- Fidelity-weighted fusion and classification to 4 classes

The dataset loader (`iemocap_dataset.py`) expects `[batch, features, time]` inputs:

- Text: `[batch, 768, T]`
- Audio: `[batch, 40, T]`
- Video: `[batch, 2048, T]`

## Repository Structure

```
fidelity_dgcnn_final/
├── preprocess.py                 # session-wise preprocessing CLI
├── train_model.py                # training CLI
├── iemocap_dataset.py            # dataset loader for .npz
├── models_fidelity_dgcnn.py      # fidelity-aware DGCNN (kept unmodified)
│
├── preprocessing/
│   ├── config.yaml               # preprocessing config
│   └── modules/
│       ├── text_features.py      # BERT extraction (mean-pooled + L2)
│       ├── audio_features.py     # MFCC+Δ+ΔΔ → PCA→40, z-norm, T frames
│       ├── video_features.py     # ResNet-50 face features, L2, T frames
│       └── utils.py              # label parsing and I/O
│
├── processed_chunks/             # output placeholder
├── checkpoints/                  # checkpoints placeholder
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## Installation

```
pip install -r requirements.txt
```

## Usage

Preprocess a session (auto GPU detection):

```
python preprocess.py --sessions Session1 --output_dir ./processed_chunks/
```

Fit global audio PCA (once), then preprocess:

```
python preprocess.py --sessions Session1 Session2 Session3 Session4 Session5 --fit_pca
python preprocess.py --sessions Session1 --output_dir ./processed_chunks/
```

Train on a session directory:

```
python train_model.py --data_dir ./processed_chunks/Session1 --epochs 50 --batch_size 16 --fusion_type fidelity
```

Train on multiple sessions (merged):

```
python train_model.py --data_dir ./processed_chunks --sessions Session1 Session2 Session3
```

Cross-session 5-fold (leave-one-session-out):

```
python train_model.py --data_dir ./processed_chunks --sessions Session1 Session2 Session3 Session4 Session5 --cross_validate --folds 5
```

Multi-GPU (optional):

The training script automatically wraps the model with `torch.nn.DataParallel` when more than one CUDA device is visible.

Verification hooks (concise):

- Prints `global PCA loaded` when a saved global PCA is used or after fitting.
- Prints `projection layers output dim = 256` during model init.
- Prints `cross-validation fold i/5 starting` for each fold.

## Citation

If you use this repository, please cite the original paper:

```
@inproceedings{fidelity_dgcnn_2024,
  title   = {Fidelity-Aware Dynamic Graph CNN for Multimodal Emotion Recognition},
  author  = {Authors},
  booktitle = {Venue},
  year    = {2024}
}
```

## License

This repository is provided for research purposes. Add your license of choice here.
