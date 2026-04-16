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
        # Make label 2D: shape (1,) so batched becomes (N, 1)
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.long)
        
        if self.model_type == 'transformer':
            attention_mask_tensor = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            return text_tensor, attention_mask_tensor, label_tensor
        else:
            return text_tensor, label_tensor
        
# LSTM model
class LSTM(nn.Module):
    def __init__(self, vocab_size=25000, embedding_dim=256, hidden_dim=256, 
                 output_dim=1, n_layers=1, bidirectional=False, dropout=0.5, pad_idx=0):
        super(LSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: (batch_size, seq_len)
        
        # Embed
        embedded = self.dropout(self.embedding(text))
        # embedded: (batch_size, seq_len, embedding_dim)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Get last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Dropout and FC
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output
    
# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=25000, embedding_dim=256, hidden_dim=256, 
                 output_dim=1, n_layers=2, n_heads=8, dropout=0.1, pad_idx=0, max_len=512):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len, dropout)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Fully connected output layer
        self.fc = nn.Linear(embedding_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask=None):
        # text shape: (batch_size, seq_len)
        # attention_mask shape: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        
        # Create key_padding_mask for transformer
        # PyTorch transformer expects: True for positions to IGNORE (padding)
        # Our mask has: 1 for real tokens, 0 for padding
        # So we need to invert: key_padding_mask = (attention_mask == 0)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # Embedding
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text) * math.sqrt(self.embedding_dim)
        
        # Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Transformer encoder
        # output shape: (batch_size, seq_len, embedding_dim)
        output = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        
        # Use the [CLS] token (first token) for classification
        # cls_output shape: (batch_size, embedding_dim)
        cls_output = output[:, 0, :]
        
        # Pass through fully connected layer
        output = self.dropout(cls_output)
        output = self.fc(output)
        
        # output shape: (batch_size, output_dim)
        return output


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
    print(f"!!! load_and_preprocess_data: data_type={data_type}, model_type={model_type}", file=sys.stderr)
    
    MAX_VOCAB_SIZE = 25000
    MAX_LEN = 256
    BATCH_SIZE = 32
    
    df = pd.read_parquet(data_path)
    print(f"!!! Loaded {len(df)} rows", file=sys.stderr)
    
    if data_type == 'train':
        vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
        
        for idx in range(len(df)):
            text = df.iloc[idx]['text']
            tokens = preprocess_text(text)
            for token in tokens:
                vocab.add_word(token)
        
        vocab.build_vocab()
        
        dataset = IMDBDataset(df, vocab, MAX_LEN, True, model_type)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f"!!! Returning train: dataloader + vocab", file=sys.stderr)
        return dataloader, vocab
        
    else:
        vocab = shared_vocab
        dataset = IMDBDataset(df, vocab, MAX_LEN, False, model_type)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"!!! Returning test: dataloader only", file=sys.stderr)
        return dataloader 


def train(model, iterator, optimizer, criterion, device, model_type='lstm'):
    pass

def evaluate(model, iterator, criterion, device, model_type='lstm'):
    pass

def main():
    pass

if __name__ == "__main__":
    main()
