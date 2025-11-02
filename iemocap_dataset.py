"""
IEMOCAP Dataset Loader for Preprocessed .npz Files
Loads text, audio, video features and labels from preprocessed chunks
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os


class IEMOCAPPreprocessedDataset(Dataset):
    """
    Dataset for loading preprocessed IEMOCAP data from .npz files
    
    Expected data format per file:
        text: (20, 768) - BERT features
        audio: (20, 40) - MFCC features
        video: (20, 2048) - ResNet-50 features
        label: str - emotion label (angry/happy/sad/neutral)
    """
    
    def __init__(self, data_dir, sessions=None, transform=None):
        """
        Args:
            data_dir: Path to processed_chunks directory
            sessions: List of sessions to load (e.g., ['Session1', 'Session2'])
                     If None, loads all available sessions
            transform: Optional transform to be applied on samples
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Emotion label mapping
        self.label_map = {
            'angry': 0,
            'happy': 1,
            'sad': 2,
            'neutral': 3
        }
        self.num_classes = len(self.label_map)
        
        # Load all .npz files from specified sessions
        self.samples = []
        self.labels = []
        
        if sessions is None:
            # Find all session directories
            sessions = [d.name for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('Session')]
            sessions.sort()
        
        print(f"Loading data from sessions: {sessions}")
        
        for session in sessions:
            session_path = self.data_dir / session
            if not session_path.exists():
                print(f"Warning: Session path not found: {session_path}")
                continue
            
            # Find all .npz files
            npz_files = list(session_path.glob('*.npz'))
            print(f"  {session}: Found {len(npz_files)} files")
            
            for npz_file in npz_files:
                try:
                    # Load to verify it's valid
                    data = np.load(npz_file, allow_pickle=True)
                    label = str(data['label'])
                    
                    if label in self.label_map:
                        self.samples.append(str(npz_file))
                        self.labels.append(self.label_map[label])
                    else:
                        print(f"Warning: Unknown label '{label}' in {npz_file.name}")
                        
                except Exception as e:
                    print(f"Error loading {npz_file.name}: {e}")
        
        print(f"\nTotal samples loaded: {len(self.samples)}")
        
        # Calculate label distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nLabel distribution:")
        for label_name, label_idx in self.label_map.items():
            count = counts[unique == label_idx][0] if label_idx in unique else 0
            print(f"  {label_name:>8}: {count:>4} ({count/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            text: torch.Tensor of shape (768, 20) - BERT features [features, time]
            audio: torch.Tensor of shape (40, 20) - MFCC features [features, time]
            video: torch.Tensor of shape (2048, 20) - ResNet features [features, time]
            label: int - emotion label index
        """
        # Load data
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract features - originally (20, features)
        text = data['text']    # (20, 768)
        audio = data['audio']  # (20, 40)
        video = data['video']  # (20, 2048)
        label = self.labels[idx]
        
        # Convert to tensors and transpose to (features, time) for DGCNN
        # DGCNN expects shape [batch, features, num_points]
        text = torch.from_numpy(text).float().transpose(0, 1)    # (768, 20)
        audio = torch.from_numpy(audio).float().transpose(0, 1)  # (40, 20)
        video = torch.from_numpy(video).float().transpose(0, 1)  # (2048, 20)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            text, audio, video = self.transform(text, audio, video)
        
        return text, audio, video, label
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced data
        Returns: torch.Tensor of shape (num_classes,)
        """
        label_counts = np.bincount(self.labels, minlength=self.num_classes)
        total = len(self.labels)
        weights = total / (self.num_classes * label_counts)
        return torch.from_numpy(weights).float()


def test_dataset():
    """Test the dataset loader"""
    print("="*70)
    print("Testing IEMOCAP Preprocessed Dataset")
    print("="*70)
    
    # Create dataset
    dataset = IEMOCAPPreprocessedDataset(
        data_dir='./processed_chunks',
        sessions=['Session1']
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading a sample
    print("\nTesting sample loading...")
    text, audio, video, label = dataset[0]
    
    print(f"Text shape: {text.shape} (expected: [768, 20])")
    print(f"Audio shape: {audio.shape} (expected: [40, 20])")
    print(f"Video shape: {video.shape} (expected: [2048, 20])")
    print(f"Label: {label} (type: {type(label)})")
    
    # Test batch loading with DataLoader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    for batch_text, batch_audio, batch_video, batch_labels in loader:
        print(f"Batch text shape: {batch_text.shape}")
        print(f"Batch audio shape: {batch_audio.shape}")
        print(f"Batch video shape: {batch_video.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break
    
    # Calculate class weights
    print("\nClass weights:")
    weights = dataset.get_class_weights()
    for label_name, label_idx in dataset.label_map.items():
        print(f"  {label_name:>8}: {weights[label_idx]:.4f}")
    
    print("\n" + "="*70)
    print("Dataset test completed successfully!")
    print("="*70)


if __name__ == '__main__':
    test_dataset()
