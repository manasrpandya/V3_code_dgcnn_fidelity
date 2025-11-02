#!/usr/bin/env python3
"""
Preprocess IEMOCAP sessions into per-utterance multimodal features.
Implements the paper's pipelines:
- Text: bert-base-uncased, last hidden states, mean pool, L2 normalize, replicate to T
- Audio: 16kHz, 25ms/10ms, 40 MFCC + delta + delta-delta -> PCA to 40, per-utt z-norm, resample to T
- Video: up to 120 frames uniformly with Haar-cascade face crop, ResNet-50 (PyTorch weights),
         per-frame L2 norm, temporal average to T (bin averaging)

Output per .npz contains keys:
  text: (T, 768)
  audio: (T, 40)
  video: (T, 2048)
  label: str  # preserved for compatibility with dataset loader
  utt: str
  session: str
  label_idx: int  # convenience
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
import librosa

from preprocessing.modules.text_features import TextFeatureExtractor
from preprocessing.modules.audio_features import AudioFeatureExtractor
from preprocessing.modules.video_features import VideoFeatureExtractor
from preprocessing.modules.utils import (
    parse_emotion_labels,
    find_utterance_files,
    read_transcript,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_config(cfg_path: str) -> Dict:
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def detect_device(user_device: str | None) -> str:
    try:
        import torch
        if user_device in {"cuda", "cpu"}:
            return user_device
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def save_npz(output_dir: Path, utt: str, session: str,
             text: np.ndarray, audio: np.ndarray, video: np.ndarray,
             label_str: str, label_idx: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{utt}.npz"
    np.savez_compressed(
        out_path,
        text=text.astype(np.float32),
        audio=audio.astype(np.float32),
        video=video.astype(np.float32),
        label=label_str,
        utt=utt,
        session=session,
        label_idx=np.int64(label_idx),
    )


def fit_global_pca_audio(base_dir: Path, sessions: list[str], sample_rate: int, n_mfcc: int,
                         hop_length: int, n_fft: int, output_pkl: Path, T_sample_per_utt: int = 100) -> None:
    """Fit a single global PCA on 120-d MFCC+Δ+ΔΔ across specified sessions and save to pickle."""
    X_list = []
    # Iterate sessions and utterances, same label parsing logic as main loop
    cfg_path = Path(__file__).parent / 'preprocessing' / 'config.yaml'
    cfg = load_config(str(cfg_path)) if cfg_path.exists() else {}
    label_cfg = cfg.get('preprocessing', {}).get('labels', {}) or cfg.get('labels', {})
    target_emotions = label_cfg.get('target_emotions', ['angry', 'happy', 'sad', 'neutral'])
    emotion_mapping = label_cfg.get('emotion_mapping', {
        'exc': 'happy', 'hap': 'happy', 'happy': 'happy',
        'ang': 'angry', 'angry': 'angry',
        'sad': 'sad',
        'neu': 'neutral', 'neutral': 'neutral',
    })

    for session in sessions:
        session_path = base_dir / session
        labels = parse_emotion_labels(str(session_path), emotion_mapping, target_emotions, verbose=False)
        if not labels:
            continue
        for utt in labels.keys():
            files = find_utterance_files(str(session_path), utt)
            audio_path = files.get('audio')
            if audio_path is None:
                continue
            try:
                y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                if len(y) == 0:
                    continue
                # base MFCCs (n_mfcc x frames)
                mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc,
                                             hop_length=hop_length, n_fft=n_fft)
                delta = librosa.feature.delta(mfccs, order=1)
                delta2 = librosa.feature.delta(mfccs, order=2)
                feat_120 = np.vstack([mfccs, delta, delta2]).T  # (frames, 3*n_mfcc)
                if feat_120.shape[0] == 0:
                    continue
                # uniform sample up to T_sample_per_utt frames to limit memory
                n_frames = feat_120.shape[0]
                take = min(T_sample_per_utt, n_frames)
                idx = np.linspace(0, n_frames - 1, take).astype(int)
                X_list.append(feat_120[idx])
            except Exception:
                continue
    if not X_list:
        raise RuntimeError("no audio frames collected for global PCA fit")
    X = np.vstack(X_list).astype(np.float32)
    pca = PCA(n_components=n_mfcc, svd_solver='auto', whiten=False)
    pca.fit(X)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump({'pca': pca}, f)
    print("global PCA loaded")  # confirm per requirement after fit completes


def main():
    parser = argparse.ArgumentParser(description='IEMOCAP session-wise preprocessing')
    parser.add_argument('--sessions', nargs='+', required=True,
                        help='Session names to process (e.g., Session1 Session2)')
    parser.add_argument('--output_dir', type=str, default='./processed_chunks/',
                        help='Output base directory for processed chunks')
    parser.add_argument('--base_dir', type=str, default='./IEMOCAP_full_release/IEMOCAP_full_release',
                        help='Base IEMOCAP dataset directory (containing Session1..Session5)')
    parser.add_argument('--seq_len', type=int, default=20, help='Temporal sequence length T')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--device', type=str, default=None, choices=[None, 'cuda', 'cpu'],
                        help='Device override (auto-detect if omitted)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--fit_pca', action='store_true', help='Fit global PCA for audio before preprocessing')

    args = parser.parse_args()

    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration defaults
    cfg_path = Path(__file__).parent / 'preprocessing' / 'config.yaml'
    cfg = load_config(str(cfg_path)) if cfg_path.exists() else {}

    base_dir = Path(args.base_dir or cfg.get('dataset', {}).get('base_path', './IEMOCAP_full_release/IEMOCAP_full_release'))
    output_base = Path(args.output_dir or cfg.get('dataset', {}).get('output_path', './processed_chunks'))
    T = args.seq_len or cfg.get('preprocessing', {}).get('sequence_length', 20)
    device = detect_device(args.device)

    # Label mapping
    label_cfg = cfg.get('preprocessing', {}).get('labels', {}) or cfg.get('labels', {})
    target_emotions = label_cfg.get('target_emotions', ['angry', 'happy', 'sad', 'neutral'])
    emotion_mapping = label_cfg.get('emotion_mapping', {
        'exc': 'happy', 'hap': 'happy', 'happy': 'happy',
        'ang': 'angry', 'angry': 'angry',
        'sad': 'sad',
        'neu': 'neutral', 'neutral': 'neutral',
        'oth': None, 'xxx': None,
    })
    label_to_idx = {e: i for i, e in enumerate(target_emotions)}

    logging.info('preprocessing configuration:')
    logging.info(f'  base_dir: {base_dir}')
    logging.info(f'  output_dir: {output_base}')
    logging.info(f'  sessions: {args.sessions}')
    logging.info(f'  seq_len: {T}')
    logging.info(f'  device: {device}')

    # Initialize feature extractors
    logging.info('initializing feature extractors...')
    text_extractor = TextFeatureExtractor(
        model_name=cfg.get('preprocessing', {}).get('text', {}).get('model_name', 'bert-base-uncased'),
        max_length=cfg.get('preprocessing', {}).get('text', {}).get('max_length', 512),
        device=device,
        verbose=args.verbose,
    )
    audio_extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate,
        n_mfcc=cfg.get('preprocessing', {}).get('audio', {}).get('n_mfcc', 40),
        hop_length=cfg.get('preprocessing', {}).get('audio', {}).get('hop_length', 160),
        n_fft=cfg.get('preprocessing', {}).get('audio', {}).get('n_fft', 400),
        normalize=True,
        verbose=args.verbose,
    )
    vid_cfg = cfg.get('preprocessing', {}).get('video', {})
    video_extractor = VideoFeatureExtractor(
        face_size=tuple(vid_cfg.get('face_size', [224, 224])),
        device=device,
        weights_url=vid_cfg.get('resnet_weights_url', 'https://download.pytorch.org/models/resnet50-0676ba61.pth'),
        cascade_file=vid_cfg.get('cascade_file', 'haarcascade_frontalface_default.xml'),
        verbose=args.verbose,
    )

    # Optional: fit global PCA
    pca_model = None
    pca_path = Path(__file__).parent / 'preprocessing' / 'global_pca_audio.pkl'
    if args.fit_pca:
        try:
            fit_global_pca_audio(
                base_dir=base_dir,
                sessions=args.sessions,
                sample_rate=args.sample_rate,
                n_mfcc=cfg.get('preprocessing', {}).get('audio', {}).get('n_mfcc', 40),
                hop_length=cfg.get('preprocessing', {}).get('audio', {}).get('hop_length', 160),
                n_fft=cfg.get('preprocessing', {}).get('audio', {}).get('n_fft', 400),
                output_pkl=pca_path,
            )
        except Exception as e:
            logging.error(f'failed to fit global PCA: {e}')

    # Load global PCA if available
    if pca_path.exists():
        try:
            with open(pca_path, 'rb') as f:
                obj = pickle.load(f)
                pca_model = obj.get('pca', None)
            if pca_model is not None:
                print("global PCA loaded")
        except Exception:
            pca_model = None

    # Process each session
    for session in args.sessions:
        session_path = base_dir / session
        if not session_path.exists():
            logging.warning(f'session not found: {session_path}')
            continue

        logging.info(f'parsing labels for {session}...')
        labels = parse_emotion_labels(str(session_path), emotion_mapping, target_emotions, verbose=args.verbose)
        if len(labels) == 0:
            logging.warning(f'no labeled utterances found for {session}')
            continue

        utt_ids = sorted(labels.keys())
        session_out_dir = output_base / session
        session_out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(utt_ids, desc=f'{session} utterances')
        saved, skipped = 0, 0
        for utt in pbar:
            lab = labels[utt]
            lab_idx = label_to_idx.get(lab, -1)

            files = find_utterance_files(str(session_path), utt)
            audio_path = files.get('audio')
            text_ref = files.get('text')
            video_path = files.get('video')

            # Ensure all modalities exist per paper requirement
            if audio_path is None or text_ref is None or video_path is None:
                skipped += 1
                continue

            # Read transcript (handles per-utt or dialog transcription)
            transcript = read_transcript(str(session_path), utt)

            # Extract features
            try:
                text_feat = text_extractor.extract_features(transcript, target_length=T)
            except Exception:
                text_feat = np.zeros((T, 768), dtype=np.float32)

            try:
                audio_feat = audio_extractor.extract_features(audio_path, target_length=T, pca_model=pca_model)
            except Exception:
                audio_feat = np.zeros((T, 40), dtype=np.float32)

            try:
                video_feat = video_extractor.extract_features(video_path, utt, target_length=T)
            except Exception:
                video_feat = np.zeros((T, 2048), dtype=np.float32)

            # Save
            save_npz(session_out_dir, utt, session, text_feat, audio_feat, video_feat, lab, lab_idx)
            saved += 1

            if saved % 100 == 0:
                pbar.set_postfix(saved=saved, skipped=skipped)

        logging.info(f'{session}: saved={saved}, skipped={skipped}, out={session_out_dir}')

    logging.info('preprocessing completed.')


if __name__ == '__main__':
    main()
