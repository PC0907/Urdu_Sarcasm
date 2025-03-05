import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Function to load and preprocess dataset with optional EDA
def load_and_preprocess_dataset(file_path, explore=True):
    try:
        df = pd.read_csv(file_path, sep='\t')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    
    # Check required columns
    required_columns = ['text', 'classify']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    
    # Convert classify column to numeric first
    df['classify'] = pd.to_numeric(df['classify'], errors='coerce')
    
    # Handle missing values in classify column
    initial_count = len(df)
    df = df.dropna(subset=['classify'])
    new_count = len(df)
    
    if new_count < initial_count:
        print(f"Removed {initial_count - new_count} rows with missing 'classify' values")
    
    # Validate classification values
    valid_labels = {0, 1}
    invalid_mask = ~df['classify'].isin(valid_labels)
    
    if invalid_mask.any():
        print(f"Found {invalid_mask.sum()} invalid label(s). Examples:")
        print(df[invalid_mask].head())
        df = df[~invalid_mask]
    
    # Convert to integer after cleaning
    df['label'] = df['classify'].astype(int)
    
    # Handle text NaNs
    df = df.dropna(subset=['text'])
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    if explore:
        print(f"\nFinal dataset shape: {df.shape}")
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        print("\nNull values count:")
        print(df.isnull().sum())
        print("\nData types:")
        print(df.dtypes)
        
        df['text_length'] = df['text'].apply(len)
        print("\nText length statistics:")
        print(df['text_length'].describe())
        print("\nClass distribution:")
        print(df['label'].value_counts())
        
        # Visualizations
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'])
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.show()
        
        plt.figure(figsize=(8, 5))
        sns.countplot(x='label', data=df)
        plt.title('Class Distribution')
        plt.show()
    
    return df

# Custom Dataset class remains the same
class UrduSarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Rest of the functions (model, training, evaluation) remain unchanged

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train Urdu Sarcasm Detection Model')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training dataset TSV file')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset TSV file')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size (e.g., 0.1 for 10%)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load datasets
    print("Loading training dataset...")
    train_df = load_and_preprocess_dataset(args.train_path, explore=True)
    print("\nLoading test dataset...")
    test_df = load_and_preprocess_dataset(args.test_path, explore=True)
    
    # Split training data into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].values,
        train_df['label'].values,
        test_size=args.val_size,
        random_state=42,
        stratify=train_df['label'].values
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Create datasets
    train_dataset = UrduSarcasmDataset(train_texts, train_labels, tokenizer)
    val_dataset = UrduSarcasmDataset(val_texts, val_labels, tokenizer)
    test_dataset = UrduSarcasmDataset(test_df['text'].values, test_df['label'].values, tokenizer)
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = mBERTBiGRUMHA(num_classes=2)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50,
        patience=5,
        delta=0.001
    )
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Test the model
    test_loss, test_accuracy, test_f1, test_precision, test_recall = test_model(model, test_loader, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }, 'urdu_sarcasm_model.pt')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
