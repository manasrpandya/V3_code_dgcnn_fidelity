"""
Text feature extraction using BERT
Extracts 768-dimensional embeddings from utterance transcripts
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class TextFeatureExtractor:
    """
    Extract BERT-based text features from transcripts.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 max_length: int = 512,
                 device: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize BERT model and tokenizer.
        
        Args:
            model_name: Pretrained BERT model name
            max_length: Maximum sequence length for tokenization
            device: Device to run on ('cuda' or 'cpu')
            verbose: Print initialization info
        """
        self.model_name = model_name
        self.max_length = max_length
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if verbose:
            print(f"Initializing BERT model: {model_name}")
            print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if verbose:
                print(f"BERT model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {e}")
    
    def extract_features(self, text: str, target_length: int = 20) -> np.ndarray:
        """
        Extract BERT features from text and average to target temporal length.
        
        Args:
            text: Input transcript text
            target_length: Target temporal sequence length (T)
            
        Returns:
            Features of shape (target_length, 768)
        """
        if not text or text.strip() == "":
            # Return zero features for empty text
            return np.zeros((target_length, 768), dtype=np.float32)
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state: (batch_size=1, seq_len, hidden_size=768)
                hidden_states = outputs.last_hidden_state
            
            # Average over sequence dimension to get (1, 768)
            # Exclude [CLS] and [SEP] tokens
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            # Masked average
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_hidden = sum_hidden / sum_mask
            
            # L2 normalize and convert to numpy
            norm = torch.norm(mean_hidden, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            mean_hidden = mean_hidden / norm
            features = mean_hidden.cpu().numpy().squeeze()  # (768,)
            
            # Replicate to target_length: (target_length, 768)
            # This matches the temporal dimension of audio and video
            features_temporal = np.tile(features, (target_length, 1))
            
            return features_temporal.astype(np.float32)
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting features from text: {e}")
            return np.zeros((target_length, 768), dtype=np.float32)
    
    def extract_batch(self, texts: list, target_length: int = 20) -> np.ndarray:
        """
        Extract features from a batch of texts.
        
        Args:
            texts: List of transcript texts
            target_length: Target temporal sequence length
            
        Returns:
            Features of shape (batch_size, target_length, 768)
        """
        if not texts:
            return np.array([])
        
        try:
            # Filter out empty texts
            valid_texts = [t if t and t.strip() else "[EMPTY]" for t in texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
            
            # Average over sequence dimension
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_hidden = sum_hidden / sum_mask  # (batch, 768)
            
            # L2 normalize and convert to numpy
            norm = torch.norm(mean_hidden, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            mean_hidden = mean_hidden / norm
            features = mean_hidden.cpu().numpy()  # (batch, 768)
            
            # Replicate to target_length for each sample
            batch_size = features.shape[0]
            features_temporal = np.tile(features[:, np.newaxis, :], (1, target_length, 1))
            
            return features_temporal.astype(np.float32)
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting batch features: {e}")
            return np.zeros((len(texts), target_length, 768), dtype=np.float32)
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache()
