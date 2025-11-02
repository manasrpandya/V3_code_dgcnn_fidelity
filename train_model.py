#!/usr/bin/env python3
"""
Train Fidelity-Aware DGCNN on preprocessed IEMOCAP data.
- Loads per-utterance .npz from session folders
- Merges specified sessions into one dataset
- Splits into 80/10/10 train/val/test
- Trains FidelityAwareMultimodalDGCNN
- Logs accuracy and weighted F1
- Saves best checkpoint
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from iemocap_dataset import IEMOCAPPreprocessedDataset
from models_fidelity_dgcnn import FidelityAwareMultimodalDGCNN


class HyperParams:
    def __init__(self):
        self.orig_d_l = 768
        self.orig_d_a = 40
        self.orig_d_v = 2048
        self.output_dim = 4


def detect_device(user_device: str | None) -> torch.device:
    if user_device in {"cuda", "cpu"}:
        return torch.device('cuda' if user_device == 'cuda' and torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_data(base: Path, sessions_arg: List[str] | None) -> Tuple[Path, List[str]]:
    """
    Resolve dataset base directory and sessions list.
    Accepts either:
    - base pointing to processed_chunks/ and sessions list
    - base pointing to processed_chunks/SessionX (no sessions provided)
    """
    base = base.resolve()
    if base.is_dir():
        npz_here = list(base.glob('*.npz'))
        if len(npz_here) > 0 and base.name.startswith('Session'):
            # using a single session directory directly
            return base.parent, [base.name]

    # otherwise expect base is processed_chunks and sessions provided (or autodetect)
    if sessions_arg is None or len(sessions_arg) == 0:
        sessions = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith('Session')]
        sessions.sort()
    else:
        sessions = list(sessions_arg)
    return base, sessions


def build_dataset(data_base: Path, sessions: List[str]) -> IEMOCAPPreprocessedDataset:
    return IEMOCAPPreprocessedDataset(data_dir=str(data_base), sessions=sessions)


def split_dataset(dataset, train=0.8, val=0.1, seed=42):
    n = len(dataset)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)
    return train_set, val_set, test_set


def create_loaders(train_set, val_set, test_set, batch_size: int, device: torch.device, num_workers: int = 4):
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device, desc: str):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc)
        for text, audio, video, labels in pbar:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def wrap_dataparallel_if_available(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def train_single_run(args, sessions: List[str], test_sessions: List[str] | None = None, fold_idx: int | None = None):
    device = detect_device(args.device)
    data_base, resolved_sessions = resolve_data(Path(args.data_dir), sessions)
    # datasets: train/val from train_sessions; test from test_sessions if provided
    if test_sessions is None:
        # single-run split on provided sessions
        dataset = build_dataset(data_base, resolved_sessions)
        train_set, val_set, test_set = split_dataset(dataset, train=0.8, val=0.1, seed=args.seed)
    else:
        train_dataset = build_dataset(data_base, sessions)
        test_dataset = build_dataset(data_base, test_sessions)
        # split train into train/val
        n = len(train_dataset)
        val_size = int(0.1 * n)
        train_size = n - val_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        test_set = test_dataset

    train_loader, val_loader, test_loader = create_loaders(train_set, val_set, test_set, args.batch_size, device, args.num_workers)

    hyp = HyperParams()
    model = FidelityAwareMultimodalDGCNN(hyp).to(device)
    model = wrap_dataparallel_if_available(model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_f1 = -1.0
    best_state = None
    patience = 8
    delta = 0.001
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # train epoch
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for text, audio, video, labels in pbar:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = total_loss / max(1, len(train_loader))
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        # validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch} [Val]")
        scheduler.step(val_f1)

        print(f"Epoch {epoch}/{args.epochs} :: Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")

        # save best and early stopping
        if val_f1 > best_val_f1 + delta:
            best_val_f1 = val_f1
            no_improve = 0
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            ckpt_name = f'best_model_fold{fold_idx}.pt' if fold_idx is not None else 'best_model.pt'
            ckpt_path = Path(args.checkpoint_dir) / ckpt_name
            torch.save({'epoch': epoch, 'model_state_dict': state_dict}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # evaluate on test with best checkpoint
    ckpt_name = f'best_model_fold{fold_idx}.pt' if fold_idx is not None else 'best_model.pt'
    ckpt_path = Path(args.checkpoint_dir) / ckpt_name
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device, desc="[Test]")
    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
    }
    return results


def run_cross_validation(args):
    # determine all sessions
    data_base, sessions = resolve_data(Path(args.data_dir), args.sessions)
    sessions = list(sessions)
    folds = min(args.folds, len(sessions))
    os.makedirs('./results', exist_ok=True)
    for i in range(folds):
        test_sess = [sessions[i]]
        train_sess = [s for s in sessions if s != sessions[i]]
        print(f"cross-validation fold {i+1}/{folds} starting")
        res = train_single_run(args, sessions=train_sess, test_sessions=test_sess, fold_idx=i+1)
        out_path = Path('./results') / f'fold{i+1}_metrics.json'
        with open(out_path, 'w') as f:
            import json
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fidelity-Aware DGCNN on IEMOCAP preprocessed data')
    parser.add_argument('--data_dir', type=str, default='./processed_chunks/', help='Path to processed_chunks or a specific session folder')
    parser.add_argument('--sessions', nargs='+', default=None, help='Optional: list of sessions to include, e.g., Session1 Session2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None, choices=[None, 'cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--fusion_type', type=str, default='fidelity', choices=['fidelity'])
    parser.add_argument('--cross_validate', action='store_true', help='Enable 5-fold leave-one-session-out cross-validation')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds (default 5)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cross_validate:
        run_cross_validation(args)
    else:
        # single run with provided sessions
        _, resolved = resolve_data(Path(args.data_dir), args.sessions)
        res = train_single_run(args, sessions=resolved)
