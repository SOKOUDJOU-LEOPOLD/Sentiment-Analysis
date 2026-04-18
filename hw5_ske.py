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
import argparse
import os
import json
from datetime import datetime

# NEW: Ensure NLTK resources are downloaded silently for the Autograder environment
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

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
        sorted_words = sorted(self.word_count.items(), key=lambda x: (-x[1],x[0]))
        
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

import sys
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
                 output_dim=1, n_layers=2, bidirectional=True, dropout=0.5, pad_idx=0):
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
    def __init__(self, vocab_size=25000, embedding_dim=256, hidden_dim=1024, 
                 output_dim=1, n_layers=4, n_heads=8, dropout=0.3, pad_idx=0, max_len=256 + 1):
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
            batch_first=True,
            activation='gelu' # UPDATE: GELU activation for better gradient flow
        )
        
        # Stack multiple encoder layers
        # NEW: Added Final LayerNorm for training stability
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,norm=nn.LayerNorm(embedding_dim))
        
        # Fully connected output layer
        self.fc = nn.Linear(embedding_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights() # NEW: Better weight initialization
    
    def _init_weights(self):
        # Xavier Initialization for deep Transformers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, input_ids, attention_mask=None):
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
        embedded = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        
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


# UPDATES 
# CHANGE THIS LINE:
def train(model, iterator, optimizer, criterion, device, model_type, scheduler=None, clip=1.0): # Added clip=1.0
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(iterator, desc='Training', leave=False):
        optimizer.zero_grad()
        
        if model_type == 'transformer':
            text, attention_mask, labels = batch
            text, attention_mask = text.to(device), attention_mask.to(device)
            labels = labels.to(device).float()
            predictions = model(text, attention_mask)
        else:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device).float()
            predictions = model(text)
        
        loss = criterion(predictions, labels)
        loss.backward()

        # FIXED THIS: Corrected the library call and the variable name
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == labels).float()
        acc = correct.sum() / len(labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device, model_type='lstm'): # Add model_type
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating', leave=False):
            if model_type == 'transformer':
                text, attention_mask, labels = batch
                text, attention_mask = text.to(device), attention_mask.to(device)
                labels = labels.to(device).float()
                predictions = model(text, attention_mask)
            else:
                text, labels = batch
                text = text.to(device)
                labels = labels.to(device).float()
                predictions = model(text)
            
            loss = criterion(predictions, labels)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == labels).float()
            acc = correct.sum() / len(labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Sentiment Analysis Models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'],
                        help='Model type to train: lstm or transformer')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='train.parquet',
                        help='Path to training data')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=25000,
                        help='Maximum vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional LSTM')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads (transformer only)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Maximum sequence length')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saving models and logs')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.model}_run_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"Training {args.model.upper()} Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(args.data_path)
    print(f"Total samples: {len(df)}")
    
    # ============================================================
    # CRITICAL FIX: Build vocabulary from ALL data FIRST
    # This matches what the autograder does
    # ============================================================
    print("\nBuilding vocabulary from ALL data...")
    vocab = Vocabulary(max_size=args.vocab_size)
    for idx in range(len(df)):
        text = df.iloc[idx]['text']
        tokens = preprocess_text(text)
        for token in tokens:
            vocab.add_word(token)
    vocab.build_vocab()
    print(f"Vocabulary size: {vocab.size}")
    
    # THEN split into train and validation
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets using the FULL-DATA vocabulary
    print("\nCreating datasets...")
    train_dataset = IMDBDataset(train_df, vocab, args.max_len, True, args.model)
    val_dataset = IMDBDataset(val_df, vocab, args.max_len, False, args.model)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == 'lstm':
        model = LSTM(
            vocab_size=vocab.size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.n_layers,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            pad_idx=0
        )
        model_save_name = 'lstm.pt'
    else:
        model = TransformerEncoder(
            vocab_size=vocab.size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            pad_idx=0,
            max_len=args.max_len + 1  # +1 for CLS token
        )
        model_save_name = 'transformer.pt'
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    # NEW: AdamW optimizer (better weight decay for Transformers)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()

    # NEW: OneCycleLR Scheduler (includes Warmup and Annealing)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, args.model, scheduler)
        
        # Evaluate
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, device, args.model)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {valid_loss:.4f} | Val Acc:   {valid_acc*100:.2f}%")
        
        # Save best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            
            # Save to output directory
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            
            # Save to required filename (lstm.pt or transformer.pt)
            torch.save(model.state_dict(), model_save_name)
            
            print(f"*** Best model saved! (Val Acc: {valid_acc*100:.2f}%) ***")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Loss: {best_valid_loss:.4f}")
    print(f"Best Validation Accuracy: {best_valid_acc*100:.2f}%")
    print(f"Models saved to: {args.output_dir}/")
    print(f"Final model also saved as: {model_save_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

