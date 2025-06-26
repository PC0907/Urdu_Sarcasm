import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  # Changed from BertTokenizer to support multilingual models
import pandas as pd
import numpy as np
import re

def preprocess_urdu_text(text):
    """
    Preprocess Urdu text for better tokenization and model performance.
    This function is imported from the main model file for consistency.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize Unicode characters
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Handle mixed script issues (keep Urdu, English, numbers, emojis, basic punctuation)
    # Urdu Unicode ranges: 0600-06FF, 0750-077F
    # Keep basic punctuation and emojis
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u0020-\u007E\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]', ' ', text)
    
    # Normalize repeated punctuation
    text = re.sub(r'[۔]{2,}', '۔', text)  # Urdu full stop
    text = re.sub(r'[\?]{2,}', '?', text)
    text = re.sub(r'[!]{2,}', '!', text)
    
    return text.strip()

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, preprocess_urdu=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess_urdu = preprocess_urdu

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply Urdu preprocessing if enabled
        if self.preprocess_urdu:
            text = preprocess_urdu_text(text)
        
        label = self.labels[idx]
        
        # Use the tokenizer with proper handling for multilingual models
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
            'labels': torch.tensor(label, dtype=torch.long),
            'raw_text': text  # Store preprocessed text for emoji encoder
        }

def load_urdu_dataset(file_path, has_header=True):
    """
    Load Urdu dataset in tab-separated format with columns: text, classify, Preprocessed
    
    Args:
        file_path: Path to the dataset file
        has_header: Whether the file has a header row (default: True)
    
    Returns:
        texts, labels: Lists of texts and corresponding labels
    """
    texts, labels = [], []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip header if present
            start_idx = 1 if has_header else 0
            
            for line_num, line in enumerate(lines[start_idx:], start_idx + 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Split by tab - expecting format: text\tclassify\tPreprocessed
                    parts = line.split('\t')
                    
                    if len(parts) < 2:
                        print(f"Warning: Skipping malformed line {line_num}: not enough columns")
                        continue
                    
                    text = parts[0].strip()  # First column is text
                    label_str = parts[1].strip()  # Second column is classify
                    
                    # Handle float labels (1.0, 0.0) and convert to int
                    try:
                        label_float = float(label_str)
                        label = int(label_float)
                        
                        if label not in [0, 1]:
                            print(f"Warning: Invalid label '{label}' at line {line_num}, skipping...")
                            continue
                            
                    except ValueError:
                        print(f"Warning: Cannot convert label '{label_str}' to number at line {line_num}, skipping...")
                        continue
                    
                    # Skip empty texts
                    if not text:
                        print(f"Warning: Empty text at line {line_num}, skipping...")
                        continue
                    
                    texts.append(text)
                    labels.append(label)
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    print(f"Line content: {line[:100]}...")
                    continue
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except UnicodeDecodeError as e:
        print(f"Warning: Unicode decode error in {file_path}: {e}")
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
                start_idx = 1 if has_header else 0
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            text = parts[0].strip()
                            label = int(float(parts[1].strip()))
                            if text and label in [0, 1]:
                                texts.append(text)
                                labels.append(label)
                    except:
                        continue
        except Exception as e2:
            raise UnicodeDecodeError(f"Could not decode file with UTF-8 or UTF-8-sig: {e2}")
    
    if not texts:
        raise ValueError(f"No valid data found in {file_path}")
    
    print(f"Loaded {len(texts)} samples from {file_path}")
    return texts, labels

def prepare_bert_data(train_path, test_path, batch_size=16, val_split=0.1, 
                     model_name='xlm-roberta-base', max_length=128):
    """
    Prepare data loaders for Urdu sarcasm detection with multilingual model support.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        model_name: Name of the multilingual model to use
        max_length: Maximum sequence length for tokenization
    """
    
    # Initialize tokenizer based on model choice
    print(f"Initializing tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and analyze train data
    print("Loading training data...")
    train_texts, train_labels = load_urdu_dataset(train_path, has_header=True)
    
    # Calculate class weights for balanced loss
    train_labels_np = np.array(train_labels)
    class_counts = np.bincount(train_labels_np)
    total_samples = len(train_labels_np)
    
    # Avoid division by zero
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(total_samples / (len(class_counts) * count))
        else:
            class_weights.append(1.0)
    
    class_weights = torch.FloatTensor(class_weights)
    
    print("\nClass distribution in training:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples ({count/total_samples*100:.2f}%)")
    print(f"Class weights: {class_weights.tolist()}")
    
    # Load test data
    print("Loading test data...")
    test_texts, test_labels = load_urdu_dataset(test_path, has_header=True)
    
    # Show sample texts for verification
    print("\nSample training texts:")
    for i in range(min(3, len(train_texts))):
        print(f"  {i+1}: {train_texts[i][:100]}{'...' if len(train_texts[i]) > 100 else ''} -> Label: {train_labels[i]}")
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_texts)} samples")
    print(f"Test: {len(test_texts)} samples")

    # Create datasets with Urdu preprocessing
    train_dataset = SarcasmDataset(
        train_texts, train_labels, tokenizer, 
        max_length=max_length, preprocess_urdu=True
    )
    test_dataset = SarcasmDataset(
        test_texts, test_labels, tokenizer, 
        max_length=max_length, preprocess_urdu=True
    )
    
    def collate_fn(batch):
        """Custom collate function to handle batching properly."""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'raw_text': [item['raw_text'] for item in batch]  # Keep as list for emoji encoder
        }
    
    # Create validation split
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use random seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Split sizes - Train: {train_size}, Validation: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenizers
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Test tokenization on a sample
    sample_text = train_texts[0] if train_texts else "یہ ایک ٹیسٹ ہے"
    test_encoding = tokenizer.encode_plus(
        preprocess_urdu_text(sample_text),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    print(f"\nTokenization test:")
    print(f"Original text: {sample_text[:50]}...")
    print(f"Preprocessed: {preprocess_urdu_text(sample_text)[:50]}...")
    print(f"Token IDs shape: {test_encoding['input_ids'].shape}")
    print(f"Attention mask shape: {test_encoding['attention_mask'].shape}")
    
    return train_loader, val_loader, test_loader, tokenizer, class_weights

def analyze_dataset_statistics(texts, labels, dataset_name="Dataset"):
    """
    Analyze and print dataset statistics.
    """
    print(f"\n{dataset_name} Statistics:")
    print(f"Total samples: {len(texts)}")
    
    # Text length statistics
    text_lengths = [len(text.split()) for text in texts]
    print(f"Average text length: {np.mean(text_lengths):.2f} words")
    print(f"Max text length: {max(text_lengths)} words")
    print(f"Min text length: {min(text_lengths)} words")
    
    # Character count statistics  
    char_lengths = [len(text) for text in texts]
    print(f"Average character length: {np.mean(char_lengths):.2f} chars")
    
    # Label distribution
    label_counts = np.bincount(labels)
    print(f"Label distribution:")
    for i, count in enumerate(label_counts):
        print(f"  Class {i}: {count} ({count/len(labels)*100:.2f}%)")
    
    # Check for potential issues
    empty_texts = sum(1 for text in texts if not text.strip())
    if empty_texts > 0:
        print(f"Warning: {empty_texts} empty texts found!")
    
    return {
        'text_lengths': text_lengths,
        'char_lengths': char_lengths,
        'label_counts': label_counts
    }

if __name__ == "__main__":
    # Test the data loading functionality
    print("Testing Urdu data loading...")
    
    try:
        # Test with sample paths - adjust these to your actual file paths
        train_loader, val_loader, test_loader, tokenizer, class_weights = prepare_bert_data(
            'data/urdu_sarcasm/train.txt',
            'data/urdu_sarcasm/test.txt',
            batch_size=8,
            model_name='xlm-roberta-base'
        )
        
        print("\nTesting data loader...")
        batch = next(iter(train_loader))
        print("Batch keys:", batch.keys())
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Labels shape:", batch['labels'].shape)
        print("Raw text count:", len(batch['raw_text']))
        print("Sample raw text:", batch['raw_text'][0][:100])
        
        print("\nData loading successful!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please check your file paths and data format.")
