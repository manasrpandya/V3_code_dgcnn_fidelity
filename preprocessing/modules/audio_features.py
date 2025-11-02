"""
Audio feature extraction using MFCC
Produces 120-dim (40 MFCC + delta + delta-delta) per-frame features.
If a global PCA model is provided, applies it to obtain 40-dim features per frame.
Finally z-normalizes per utterance and resamples to target T.
"""

import numpy as np
import librosa
from sklearn.decomposition import PCA
import warnings
from typing import Optional

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    Extract MFCC features from audio files.
    """
    
    def __init__(self, sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 hop_length: int = 160,
                 n_fft: int = 400,
                 normalize: bool = True,
                 verbose: bool = False):
        """
        Initialize audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            hop_length: Hop length for STFT
            n_fft: FFT window size
            normalize: Whether to normalize features per utterance
            verbose: Print processing info
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalize = normalize
        self.verbose = verbose
        
        if verbose:
            print(f"Audio feature extractor initialized:")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  MFCC coefficients: {n_mfcc}")
            print(f"  Hop length: {hop_length}")
            print(f"  FFT size: {n_fft}")
    
    def extract_features(self, audio_path: str, target_length: int = 20, pca_model=None) -> np.ndarray:
        """
        Extract MFCC+Δ+ΔΔ features and optionally apply global PCA.
        
        Args:
            audio_path: Path to audio file
            target_length: Target temporal sequence length (T)
            
        Returns:
            Audio features of shape (target_length, n_mfcc) where n_mfcc=40
        """
        try:
            # Load audio file and resample to target sample rate
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            if len(audio) == 0:
                if self.verbose:
                    print(f"Warning: Empty audio file {audio_path}")
                return np.zeros((target_length, self.n_mfcc), dtype=np.float32)
            
            # Extract base MFCC features (shape: n_mfcc x frames)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Compute delta and delta-delta
            delta = librosa.feature.delta(mfccs, order=1)
            delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Stack to 3*n_mfcc features
            feat_120 = np.vstack([mfccs, delta, delta2])  # (3*n_mfcc, frames)
            
            # Transpose to (time, features)
            feat_120 = feat_120.T  # (num_frames, 3*n_mfcc)
            
            # Apply global PCA to 40 dims if provided; fallback to per-utterance PCA otherwise
            if pca_model is not None:
                try:
                    feat_pca = pca_model.transform(feat_120)
                except Exception:
                    # safety fallback if transform fails
                    feat_pca = feat_120[:, :self.n_mfcc] if feat_120.shape[1] >= self.n_mfcc else np.pad(
                        feat_120, ((0, 0), (0, self.n_mfcc - feat_120.shape[1])), mode='constant'
                    )
            else:
                # per-utterance PCA fallback to preserve backward compatibility
                try:
                    if feat_120.shape[0] >= self.n_mfcc:
                        pca = PCA(n_components=self.n_mfcc, svd_solver='auto', whiten=False)
                        feat_pca = pca.fit_transform(feat_120)
                    else:
                        n_comp = min(self.n_mfcc, feat_120.shape[0], feat_120.shape[1])
                        pca = PCA(n_components=n_comp, svd_solver='auto', whiten=False)
                        temp = pca.fit_transform(feat_120)
                        if temp.shape[1] < self.n_mfcc:
                            pad = np.zeros((temp.shape[0], self.n_mfcc - temp.shape[1]))
                            feat_pca = np.concatenate([temp, pad], axis=1)
                        else:
                            feat_pca = temp
                except Exception:
                    feat_pca = feat_120[:, :self.n_mfcc] if feat_120.shape[1] >= self.n_mfcc else np.pad(
                        feat_120, ((0, 0), (0, self.n_mfcc - feat_120.shape[1])), mode='constant'
                    )
            
            # Z-normalize per utterance if requested
            if self.normalize and feat_pca.shape[0] > 0:
                mean = np.mean(feat_pca, axis=0, keepdims=True)
                std = np.std(feat_pca, axis=0, keepdims=True)
                std = np.where(std < 1e-8, 1.0, std)
                feat_pca = (feat_pca - mean) / std
            
            # Downsample or upsample to target length
            mfccs = self._temporal_resample(feat_pca, target_length)
            
            return mfccs.astype(np.float32)
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros((target_length, self.n_mfcc), dtype=np.float32)
    
    def _temporal_resample(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample features to target temporal length.
        
        Args:
            features: Input features (T, D)
            target_length: Target length
            
        Returns:
            Resampled features (target_length, D)
        """
        if features.shape[0] == 0:
            return np.zeros((target_length, features.shape[1]))
        
        current_length = features.shape[0]
        
        if current_length == target_length:
            return features
        
        if current_length > target_length:
            # Downsample by averaging chunks
            indices = np.linspace(0, current_length, target_length + 1).astype(int)
            downsampled = []
            for i in range(target_length):
                start, end = indices[i], indices[i + 1]
                if start < end:
                    downsampled.append(features[start:end].mean(axis=0))
                else:
                    downsampled.append(features[start])
            return np.array(downsampled)
        else:
            # Upsample using linear interpolation
            indices = np.linspace(0, current_length - 1, target_length)
            resampled = np.array([
                np.interp(indices, np.arange(current_length), features[:, d])
                for d in range(features.shape[1])
            ]).T
            return resampled
    
    def extract_batch(self, audio_paths: list, target_length: int = 20) -> np.ndarray:
        """
        Extract features from a batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            target_length: Target temporal sequence length
            
        Returns:
            Features of shape (batch_size, target_length, n_mfcc)
        """
        features_list = []
        for audio_path in audio_paths:
            features = self.extract_features(audio_path, target_length)
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            if self.verbose:
                print(f"Error getting duration for {audio_path}: {e}")
            return 0.0
