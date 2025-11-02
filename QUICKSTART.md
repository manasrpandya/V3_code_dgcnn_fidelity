# Quickstart

## Environment

```
pip install -r requirements.txt
```

## Preprocess (per session)

Place IEMOCAP under `./IEMOCAP_full_release/IEMOCAP_full_release/Session*/...`.

Run preprocessing (auto GPU detection):

```
python preprocess.py --sessions Session1 --output_dir ./processed_chunks/
```

Options:

- `--seq_len`: temporal frames T (default 20)
- `--sample_rate`: audio sample rate (default 16000)
- `--device`: `cuda` or `cpu` (default auto)

Outputs written to `./processed_chunks/Session1/*.npz`.

Fit global audio PCA once (recommended), then preprocess:

```
python preprocess.py --sessions Session1 Session2 Session3 Session4 Session5 --fit_pca
python preprocess.py --sessions Session1 --output_dir ./processed_chunks/
```

## Train

Single session directory:

```
python train_model.py --data_dir ./processed_chunks/Session1 --epochs 50 --batch_size 16 --fusion_type fidelity
```

Multiple sessions (merged):

```
python train_model.py --data_dir ./processed_chunks --sessions Session1 Session2
```

Notes:

Cross-session (5-fold leave-one-session-out):

```
python train_model.py --data_dir ./processed_chunks --sessions Session1 Session2 Session3 Session4 Session5 --cross_validate --folds 5
```

Notes:

- GPU is optional; CPU is supported.
- Checkpoints saved to `./checkpoints/`.
 - Model projects each modality to a 256-d shared latent before DGCNN.
 - Multi-GPU is automatic when multiple CUDA devices are visible.
