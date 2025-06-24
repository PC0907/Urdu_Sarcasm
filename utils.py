import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import re
import argparse

class UrduSarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.avg_word_length = self.calculate_avg_word_length()
        self.avg_sentence_length = self.calculate_avg_sentence_length()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Preprocess the text for Urdu
        #text = self.preprocess_urdu_text(text)

        # Encode the text using multilingual tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    # def preprocess_urdu_text(self, text):
    #     """
    #     Preprocessing specifically designed for Urdu text with emojis
    #     """
    #     # Remove extra whitespace but preserve Urdu characters and emojis
    #     text = re.sub(r'\s+', ' ', text).strip()
        
    #     # Optional: Remove some problematic characters but keep Urdu script and emojis
    #     # Be very careful here - don't remove Urdu characters!
    #     text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u200C\u200D\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]', ' ', text)
        
    #     # Clean up multiple spaces again
    #     text = re.sub(r'\s+', ' ', text).strip()
        
    #     return text

    def calculate_avg_word_length(self):
        total_words = 0
        total_length = 0
        for text in self.texts:
            # Split by whitespace for Urdu text
            words = text.split()
            total_words += len(words)
            total_length += sum(len(word) for word in words)
        return total_length / total_words if total_words > 0 else 0

    def calculate_avg_sentence_length(self):
        total_sentences = len(self.texts)
        total_words = sum(len(text.split()) for text in self.texts)
        return total_words / total_sentences if total_sentences > 0 else 0

def prepare_urdu_bert_data(train_path, test_path, batch_size=16, model_name='bert-base-multilingual-cased'):
    """
    Prepare Urdu sarcasm data from TSV files
    """
    # Use multilingual tokenizer for Urdu support
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load train data from TSV
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path, sep='\t', encoding='utf-8')
    train_texts = train_df['text'].tolist()
    train_labels = train_df['classify'].astype(int).tolist()
    
    # Load test data from TSV
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path, sep='\t', encoding='utf-8')
    test_texts = test_df['text'].tolist()
    test_labels = test_df['classify'].astype(int).tolist()
    
    print(f"Train data size: {len(train_texts)}")
    print(f"Test data size: {len(test_texts)}")
    print("Train label distribution:")
    print(pd.Series(train_labels).value_counts())
    print("Test label distribution:")
    print(pd.Series(test_labels).value_counts())

    # Create datasets
    train_dataset = UrduSarcasmDataset(train_texts, train_labels, tokenizer)
    test_dataset = UrduSarcasmDataset(test_texts, test_labels, tokenizer)

    print(f"Average word length in train dataset: {train_dataset.avg_word_length:.2f}")
    print(f"Average sentence length in train dataset: {train_dataset.avg_sentence_length:.2f}")
    print(f"Average word length in test dataset: {test_dataset.avg_word_length:.2f}")
    print(f"Average sentence length in test dataset: {test_dataset.avg_sentence_length:.2f}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, tokenizer

# Keep original function for English data
def prepare_bert_data(train_path, test_path, batch_size=16):
    """Original function for English data"""
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
    # Load train data
    train_texts, train_labels = [], []
    with open(train_path, 'r') as f:
        for line in f:
            text, label = line.rsplit(' ', 1)
            train_texts.append(text)
            train_labels.append(int(label))
    
    # Load test data
    test_texts, test_labels = [], []
    with open(test_path, 'r') as f:
        for line in f:
            text, label = line.rsplit(' ', 1)
            test_texts.append(text)
            test_labels.append(int(label))

    # Create datasets (using original SarcasmDataset for English)
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
    test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer

class SarcasmDataset(Dataset):
    """Original dataset class for English data"""
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

        # English preprocessing
        text = self.preprocess_text(text)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def preprocess_text(self, text):
        # English preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def evaluate(model, test_loader, criterion, device, zero_division=0):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=zero_division)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=zero_division)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=zero_division)
    
    return avg_loss, accuracy, f1, precision, recall
