"""
Utility functions for IEMOCAP preprocessing
Handles label parsing, utterance indexing, and I/O operations
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


def parse_emotion_labels(session_path: str, emotion_mapping: Dict[str, str], 
                         target_emotions: List[str], verbose: bool = False) -> Dict[str, str]:
    """
    Parse emotion labels from IEMOCAP EmoEvaluation files.
    
    Args:
        session_path: Path to session directory (e.g., Session1)
        emotion_mapping: Dictionary mapping source emotions to target emotions
        target_emotions: List of emotions to keep
        verbose: Print parsing information
        
    Returns:
        Dictionary mapping utterance_id to emotion label
    """
    labels = {}
    evaluation_dir = os.path.join(session_path, "dialog", "EmoEvaluation")
    
    if not os.path.exists(evaluation_dir):
        print(f"Warning: EmoEvaluation directory not found at {evaluation_dir}")
        return labels
    
    # Find all evaluation files, skip files starting with "._"
    eval_files = []
    for f in glob.glob(os.path.join(evaluation_dir, "*.txt")):
        basename = os.path.basename(f)
        if not basename.startswith("._"):
            eval_files.append(f)
    
    if verbose:
        print(f"Found {len(eval_files)} evaluation files in {evaluation_dir}")
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%') or line.startswith('C-') or line.startswith('A-'):
                        continue
                    
                    # Parse line format: [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
                    # Example: [6.2901 - 8.2357]       Ses01F_impro01_F000     neu     [2.5000, 2.5000, 2.5000]
                    # Fields are separated by multiple spaces/tabs
                    
                    # Check if line starts with timestamp
                    if not line.startswith('['):
                        continue
                    
                    # Split by whitespace
                    parts = line.split()
                    if len(parts) >= 4:
                        # parts[0-2]: timestamp [START_TIME - END_TIME]
                        # parts[3]: utterance_id
                        # parts[4]: emotion
                        # parts[5+]: VAD values
                        
                        utterance_id = parts[3].strip()
                        emotion = parts[4].strip().lower()
                        
                        # Apply emotion mapping (e.g., excited -> happy)
                        if emotion in emotion_mapping:
                            emotion = emotion_mapping[emotion]
                            # Skip if mapped to None/null
                            if emotion is None:
                                continue
                        
                        # Keep only target emotions
                        if emotion in target_emotions:
                            labels[utterance_id] = emotion
        except Exception as e:
            if verbose:
                print(f"Error parsing {eval_file}: {e}")
    
    if verbose:
        print(f"Parsed {len(labels)} utterances with target emotions")
        # Print emotion distribution
        emotion_counts = defaultdict(int)
        for emotion in labels.values():
            emotion_counts[emotion] += 1
        print("Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}")
    
    return labels


def find_utterance_files(session_path: str, utterance_id: str) -> Dict[str, Optional[str]]:
    """
    Find all modality files for a given utterance ID.
    
    Args:
        session_path: Path to session directory
        utterance_id: Utterance ID (e.g., Ses01F_impro01_F000)
        
    Returns:
        Dictionary with keys 'audio', 'text', 'video' pointing to file paths
    """
    files = {
        'audio': None,
        'text': None,
        'video': None
    }
    
    # Extract session and dialog info from utterance_id
    # Format: Ses01F_impro01_F000
    match = re.match(r'(Ses\d+[MF])_(.*?)_[MF]\d+', utterance_id)
    if not match:
        return files
    
    session_id = match.group(1)
    dialog_id = f"{session_id}_{match.group(2)}"
    
    # Audio file: sentences/wav/<dialog_id>/<utterance_id>.wav
    audio_path = os.path.join(session_path, "sentences", "wav", dialog_id, f"{utterance_id}.wav")
    if os.path.exists(audio_path):
        files['audio'] = audio_path
    
    # Text file: sentences/txt/<dialog_id>/<utterance_id>.txt (preferred)
    text_path = os.path.join(session_path, "sentences", "txt", dialog_id, f"{utterance_id}.txt")
    if os.path.exists(text_path):
        files['text'] = text_path
    else:
        # Fallback: dialog-level transcription file
        transcription_file = os.path.join(session_path, "dialog", "transcriptions", f"{dialog_id}.txt")
        if os.path.exists(transcription_file):
            files['text'] = transcription_file
    
    # Video file: dialog/avi/<dialog_id>.avi or dialog/avi/DivX/<dialog_id>.avi
    video_path = os.path.join(session_path, "dialog", "avi", f"{dialog_id}.avi")
    if os.path.exists(video_path):
        files['video'] = video_path
    else:
        # Try DivX subdirectory
        video_path_divx = os.path.join(session_path, "dialog", "avi", "DivX", f"{dialog_id}.avi")
        if os.path.exists(video_path_divx):
            files['video'] = video_path_divx
    
    return files


def read_transcript(session_path: str, utterance_id: str) -> str:
    """
    Read transcript for a specific utterance from dialog-level transcription files.
    
    Args:
        session_path: Path to session directory
        utterance_id: Utterance ID (e.g., Ses01F_impro01_F000)
        
    Returns:
        Transcript text
    """
    try:
        # Extract dialog ID from utterance ID
        # Format: Ses01F_impro01_F000 -> Ses01F_impro01
        match = re.match(r'(Ses\d+[MF]_.*?)_[MF]\d+', utterance_id)
        if not match:
            return ""
        
        dialog_id = match.group(1)
        # Preferred: utterance-level transcript file
        text_path = os.path.join(session_path, "sentences", "txt", dialog_id, f"{utterance_id}.txt")
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                pass
        
        # Fallback: dialog-level transcription file
        transcription_file = os.path.join(session_path, "dialog", "transcriptions", f"{dialog_id}.txt")
        if os.path.exists(transcription_file):
            with open(transcription_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(utterance_id):
                        if ':' in line:
                            transcript = line.split(':', 1)[1].strip()
                            return transcript
        
        return ""
        
    except Exception as e:
        return ""


def save_utterance(output_dir: str, utterance_id: str, text_feat: np.ndarray,
                   audio_feat: np.ndarray, video_feat: np.ndarray, label: str):
    """
    Save preprocessed utterance features to disk.
    
    Args:
        output_dir: Output directory for session
        utterance_id: Utterance ID
        text_feat: Text features (T, 768)
        audio_feat: Audio features (T, 40)
        video_feat: Video features (T, 2048)
        label: Emotion label
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{utterance_id}.npz")
    
    try:
        np.savez_compressed(
            output_file,
            text=text_feat,
            audio=audio_feat,
            video=video_feat,
            label=label
        )
    except Exception as e:
        print(f"Error saving {output_file}: {e}")


def load_utterance(npz_file: str) -> Dict[str, any]:
    """
    Load preprocessed utterance from disk.
    
    Args:
        npz_file: Path to .npz file
        
    Returns:
        Dictionary with keys 'text', 'audio', 'video', 'label'
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        return {
            'text': data['text'],
            'audio': data['audio'],
            'video': data['video'],
            'label': str(data['label'])
        }
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None


def get_session_path(base_path: str, session_name: str) -> str:
    """
    Get full path to a session directory.
    
    Args:
        base_path: Base IEMOCAP directory
        session_name: Session name (e.g., Session1)
        
    Returns:
        Full path to session
    """
    return os.path.join(base_path, session_name)


def create_output_directory(output_base: str, session_name: str) -> str:
    """
    Create output directory for a session.
    
    Args:
        output_base: Base output directory
        session_name: Session name
        
    Returns:
        Path to session output directory
    """
    output_dir = os.path.join(output_base, session_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def temporal_downsample(features: np.ndarray, target_length: int) -> np.ndarray:
    """
    Downsample or upsample features to target temporal length.
    Uses simple averaging for downsampling and linear interpolation for upsampling.
    
    Args:
        features: Input features (T, D)
        target_length: Target temporal length
        
    Returns:
        Resampled features (target_length, D)
    """
    if len(features) == 0:
        return np.zeros((target_length, features.shape[1] if len(features.shape) > 1 else 1))
    
    current_length = features.shape[0]
    
    if current_length == target_length:
        return features
    
    if current_length > target_length:
        # Downsample by averaging
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
        if len(features.shape) == 1:
            return np.interp(indices, np.arange(current_length), features)
        else:
            return np.array([np.interp(indices, np.arange(current_length), features[:, d]) 
                           for d in range(features.shape[1])]).T
