"""
Video feature extraction using ResNet-50
Extracts 2048-dimensional features from face regions in video frames
Pure PyTorch implementation without torchvision/timm
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """ResNet-50 architecture implemented in pure PyTorch"""

    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # 2048-dim features

        if return_features:
            return features

        x = self.fc(features)
        return x


class VideoFeatureExtractor:
    """
    Extract ResNet-50 features from video frames with face detection.
    """

    def __init__(self, face_size=(224, 224),
                 device: Optional[str] = None,
                 weights_url: str = "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                 cascade_file: str = "haarcascade_frontalface_default.xml",
                 verbose: bool = False):
        """
        Initialize video feature extractor.

        Args:
            face_size: Target size for face crops
            device: Device to run on ('cuda' or 'cpu')
            weights_url: URL to download ResNet-50 pretrained weights
            cascade_file: Path to Haar cascade XML file
            verbose: Print processing info
        """
        self.face_size = face_size
        self.verbose = verbose

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if verbose:
            print(f"Initializing video feature extractor")
            print(f"Using device: {self.device}")

        # Initialize ResNet-50
        self.model = ResNet50()
        self.model.eval()

        # Load pretrained weights
        try:
            if verbose:
                print(f"Loading ResNet-50 weights from {weights_url}")
            state_dict = torch.hub.load_state_dict_from_url(weights_url, progress=verbose)
            self.model.load_state_dict(state_dict)
            if verbose:
                print("ResNet-50 weights loaded successfully")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Using randomly initialized weights")

        self.model.to(self.device)

        # Initialize face detector
        try:
            # Try to find cascade file in OpenCV data
            cascade_path = None
            if os.path.exists(cascade_file):
                cascade_path = cascade_file
            else:
                # Try OpenCV's default location
                cv2_data = cv2.data.haarcascades
                cascade_path = os.path.join(cv2_data, cascade_file)

            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                if verbose:
                    print("Warning: Could not load Haar cascade, will process full frames")
                self.face_cascade = None
            elif verbose:
                print(f"Haar cascade loaded from {cascade_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: Face detection initialization failed: {e}")
            self.face_cascade = None

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def extract_features(self, video_path: str, utterance_id: str,
                        target_length: int = 20, max_frames: int = 120) -> np.ndarray:
        """
        Extract ResNet-50 features from video file.

        Args:
            video_path: Path to video file
            utterance_id: Utterance ID to extract timing info
            target_length: Target temporal sequence length (T)

        Returns:
            Features of shape (target_length, 2048)
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if self.verbose:
                    print(f"Error: Could not open video {video_path}")
                return np.zeros((target_length, 2048), dtype=np.float32)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps == 0 or total_frames == 0:
                if self.verbose:
                    print(f"Warning: Invalid video parameters for {video_path}")
                cap.release()
                return np.zeros((target_length, 2048), dtype=np.float32)

            # Uniformly sample up to max_frames frames across the entire dialog video
            # since utterance timestamps are not precisely aligned in video
            sample_count = min(max_frames, total_frames)
            frame_indices = np.linspace(0, total_frames - 1, sample_count).astype(int)

            features_list = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret or frame is None:
                    continue

                # Detect face and extract features
                feature = self._process_frame(frame)
                if feature is not None:
                    features_list.append(feature)

            cap.release()

            if len(features_list) == 0:
                if self.verbose:
                    print(f"Warning: No valid frames extracted from {video_path}")
                return np.zeros((target_length, 2048), dtype=np.float32)

            # Stack features
            features = np.array(features_list)  # (num_frames, 2048)

            # L2 normalize per-frame feature vectors
            eps = 1e-12
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, eps)
            features = features / norms

            # Resample to target length via bin averaging
            features = self._temporal_resample(features, target_length)

            return features.astype(np.float32)

        except Exception as e:
            if self.verbose:
                print(f"Error extracting features from {video_path}: {e}")
            return np.zeros((target_length, 2048), dtype=np.float32)

    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame: detect face, crop, and extract features.

        Args:
            frame: Input frame (H, W, 3) in BGR format

        Returns:
            Feature vector (2048,) or None if processing fails
        """
        try:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    face_crop = frame_rgb[y:y+h, x:x+w]
                else:
                    # No face detected, use center crop
                    h, w = frame_rgb.shape[:2]
                    crop_size = min(h, w)
                    y = (h - crop_size) // 2
                    x = (w - crop_size) // 2
                    face_crop = frame_rgb[y:y+crop_size, x:x+crop_size]
            else:
                # No face detector, use center crop
                h, w = frame_rgb.shape[:2]
                crop_size = min(h, w)
                y = (h - crop_size) // 2
                x = (w - crop_size) // 2
                face_crop = frame_rgb[y:y+crop_size, x:x+crop_size]

            # Resize to target size
            face_resized = cv2.resize(face_crop, self.face_size)

            # Convert to tensor and normalize
            face_tensor = torch.from_numpy(face_resized).float()
            face_tensor = face_tensor.permute(2, 0, 1)  # (3, H, W)
            face_tensor = face_tensor / 255.0
            face_tensor = (face_tensor - self.mean) / self.std
            face_tensor = face_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

            # Extract features
            with torch.no_grad():
                features = self.model(face_tensor, return_features=True)

            return features.cpu().numpy().squeeze()

        except Exception as e:
            if self.verbose:
                print(f"Error processing frame: {e}")
            return None

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
            # Downsample by averaging bins (temporal averaging)
            indices = np.linspace(0, current_length, target_length + 1).astype(int)
            downsampled = []
            for i in range(target_length):
                start, end = indices[i], indices[i + 1]
                if start < end:
                    downsampled.append(features[start:end].mean(axis=0))
                else:
                    downsampled.append(features[min(start, current_length - 1)])
            return np.array(downsampled)
        else:
            # Upsample using linear interpolation
            indices = np.linspace(0, current_length - 1, target_length)
            resampled = np.array([
                np.interp(indices, np.arange(current_length), features[:, d])
                for d in range(features.shape[1])
            ]).T
            return resampled

    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache()
