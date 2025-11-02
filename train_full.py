"""
Complete Training Script for Fidelity-Aware Dynamic Graph CNN
Uses preprocessed IEMOCAP data with integrated model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from iemocap_dataset import IEMOCAPPreprocessedDataset
from models_fidelity_dgcnn import FidelityAwareMultimodalDGCNN


class HyperParams:
    """Hyperparameters for the model"""
    def __init__(self, args):
        # Feature dimensions from our preprocessed data
        self.orig_d_l = 768    # BERT text features
        self.orig_d_a = 40     # MFCC audio features
        self.orig_d_v = 2048   # ResNet-50 video features
        self.output_dim = 4    # 4 emotion classes (angry, happy, sad, neutral)
        
        # Training hyperparameters
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.device = args.device
        
        # Model hyperparameters
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.k_neighbors = args.k_neighbors


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for text, audio, video, labels in pbar:
        # Move to device
        text = text.to(device)
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(text, audio, video)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for text, audio, video, labels in pbar:
            # Move to device
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, cm, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
    }, checkpoint_path)


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n{'='*70}")
    print("Fidelity-Aware Dynamic Graph CNN Training".center(70))
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    print(f"\n{'='*70}")
    print("Loading Dataset")
    print(f"{'='*70}")
    
    sessions = args.sessions.split(',') if isinstance(args.sessions, str) else args.sessions
    dataset = IEMOCAPPreprocessedDataset(
        data_dir=args.data_dir,
        sessions=sessions
    )
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print(f"\n{'='*70}")
    print("Initializing Model")
    print(f"{'='*70}")
    
    hyp_params = HyperParams(args)
    model = FidelityAwareMultimodalDGCNN(hyp_params)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function (with class weights if specified)
    if args.use_class_weights:
        class_weights = dataset.get_class_weights().to(device)
        print(f"\nUsing class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}\n")
    
    best_acc = 0.0
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1, cm, preds, labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_checkpoint = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, best_acc, best_checkpoint)
            print(f"  ✓ Saved best model (Acc: {best_acc:.4f}, F1: {best_f1:.4f})")
        
        # Save latest model
        latest_checkpoint = os.path.join(args.checkpoint_dir, 'latest_model.pt')
        save_checkpoint(model, optimizer, epoch, val_acc, latest_checkpoint)
        
        print()
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best Validation F1 Score: {best_f1:.4f}")
    
    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, val_acc, val_f1, cm, preds, labels = validate(
        model, val_loader, criterion, device, args.epochs-1
    )
    
    print(f"\nFinal Confusion Matrix:")
    print(cm)
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    label_names = ['angry', 'happy', 'sad', 'neutral']
    print(classification_report(labels, preds, target_names=label_names, digits=4))
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"  - best_model.pt (Acc: {best_acc:.4f})")
    print(f"  - latest_model.pt")
    print(f"  - training_history.json")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fidelity-Aware DGCNN on IEMOCAP')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./processed_chunks',
                       help='Path to preprocessed data directory')
    parser.add_argument('--sessions', type=str, default='Session1',
                       help='Comma-separated list of sessions (e.g., Session1,Session2)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--k_neighbors', type=int, default=10,
                       help='Number of neighbors for dynamic graph')
    parser.add_argument('--fusion_type', type=str, default='fidelity',
                       choices=['fidelity', 'concat', 'attention'],
                       help='Fusion type (currently using fidelity)')
    parser.add_argument('--use_dgcnn_encoder', action='store_true', default=True,
                       help='Use DGCNN encoder (default: True)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
