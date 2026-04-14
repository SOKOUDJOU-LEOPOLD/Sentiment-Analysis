import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from tqdm import tqdm
import math

def preprocess_text(text):
    """
    Clean and tokenize text
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        return tokens
    return []

class Vocabulary:
    """
    Build a vocabulary from the word count
    """
    def __init__(self, max_size):
        self.max_size = max_size
        # Add <cls> token for transformer classification
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size = 3  # Start with pad, unk, and cls tokens
        
    def add_word(self, word):
        """
        Add a word to the word count disctionary
        """
        if word in self.word_count:
            self.word_count[word] += 1
        else:
            self.word_count[word] = 1        

            
    def build_vocab(self):
        """
        Build vocabulary from word counts, keeping most frequent words
        """
        # Sort words by frequency (descending)
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        
        # Add words until we reach max_size
        for word, count in sorted_words:
            if self.size >= self.max_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = self.size
                self.idx2word[self.size] = word
                self.size += 1        
                
    def text_to_indices(self, tokens, max_len, model_type='lstm'):
        """
        Convert tokens to indices with padding
        
        For LSTM: just pad to max_len
        For Transformer: add <cls> token at the beginning

        NB:
        -> in LSTM: punctuation should be converted to <unk>
        -> in Transformer: punctuation should be skipped
        """
        indices = []
        
        # For transformer, add <cls> token at the beginning
        if model_type == 'transformer':
            indices.append(self.word2idx["<cls>"])
        
        # Convert tokens to indices
        for token in tokens:
            # Check length FIRST
            if len(indices) >= max_len:
                break
            
            # Skip empty tokens
            if not token:
                continue

            # Skip punctuation-only tokens
            if model_type == "transformer":
                if all(char in string.punctuation for char in token):
                    continue
                
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<unk>"])
        
        # Pad if too short
        while len(indices) < max_len:
            indices.append(self.word2idx["<pad>"])
        
        return indices        

class IMDBDataset(Dataset):
    """
    A dataset for the IMDB dataset
    """
    def __init__(self, dataframe, vocabulary, max_len, is_training=True, model_type='lstm'):
        """
            Initialize the dataset
    
            Args:
                dataframe: pandas DataFrame with 'text' and 'label' columns
                vocabulary: Vocabulary object for converting tokens to indices
                max_len: Maximum sequence length
                is_training: Whether this is training data (not used here but could be useful)
                model_type: 'lstm' or 'transformer'

        """
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.is_training = is_training
        self.model_type = model_type
        
        # Reset index to avoid indexing issues
        self.dataframe = dataframe.reset_index(drop=True)
        
        # Preprocess all texts and store
        self.texts = []
        self.labels = []
        self.attention_masks = []
        
        for idx in range(len(self.dataframe)):
            text = self.dataframe.iloc[idx]['text']
            label = self.dataframe.iloc[idx]['label']
            
            # Tokenize text
            tokens = preprocess_text(text)
            
            # Convert to indices
            indices = vocabulary.text_to_indices(tokens, max_len, model_type)
            
            self.texts.append(indices)
            self.labels.append(label)
            
            # Always create attention mask
            attention_mask = [1 if token_idx != 0 else 0 for token_idx in indices]
            self.attention_masks.append(attention_mask)      
            
    def __len__(self):
        """Return the total number of samples"""
        return len(self.texts)        
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        For LSTM: returns (text_tensor, label_tensor)
        For Transformer: returns (text_tensor, attention_mask_tensor, label_tensor)
        """
        text_tensor = torch.tensor(self.texts[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.model_type == 'transformer':
            attention_mask_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return text_tensor, attention_mask_tensor, label_tensor
        else:
            # LSTM
            return text_tensor, label_tensor
        
# LSTM model
class LSTM(nn.Module):
    pass
    
# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    pass

# Transformer Encoder
class TransformerEncoder(nn.Module):
    pass

def load_and_preprocess_data(data_path, data_type='train', model_type='lstm', shared_vocab=None):
    """
    Load and preprocess the IMDB dataset
    
    Args:
        data_path: Path to the data files
        data_type: Type of data to load ('train' or 'test')
        model_type: Type of model ('lstm' or 'transformer')
        shared_vocab: Optional vocabulary to use (for test data)
    
    Returns:
        data_loader: DataLoader for the specified data type
        vocab: Vocabulary object (only returned for train data)
    """
    MAX_VOCAB_SIZE = 25000
    MAX_LEN = 256
    BATCH_SIZE = 32
    VAL_SPLIT = 0.1
    
    # Load data
    df = pd.read_parquet(data_path)
    
    if data_type == 'train':
        # Create vocabulary
        vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
        
        for idx in range(len(df)):
            text = df.iloc[idx]['text']
            tokens = preprocess_text(text)
            for token in tokens:
                vocab.add_word(token)
        
        vocab.build_vocab()
        
        # Split into train and validation
        train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)
        
        # Create train dataset and loader
        train_dataset = IMDBDataset(train_df, vocab, MAX_LEN, True, model_type)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Create validation dataset and loader
        val_dataset = IMDBDataset(val_df, vocab, MAX_LEN, False, model_type)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # You could return val_loader as well, or store it differently
        # For now, returning just train_loader to match expected interface
        return train_loader, vocab
        
    else:
        vocab = shared_vocab
        dataset = IMDBDataset(df, vocab, MAX_LEN, False, model_type)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        return dataloader    


def train(model, iterator, optimizer, criterion, device, model_type='lstm'):
    pass

def evaluate(model, iterator, criterion, device, model_type='lstm'):
    pass

def main():
    pass

if __name__ == "__main__":
    main()
