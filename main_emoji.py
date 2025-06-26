import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer  # Changed from BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from utils_emoji import SarcasmDataset, prepare_bert_data
import os
import numpy as np
from typing import List, Optional
import gensim.models as gsm
from pathlib import Path
import argparse
import time
import re
from utils_emoji import prepare_bert_data

class EmojiEncoder(nn.Module):
    """Encodes emoji sequences into fixed-dimensional embeddings."""
    
    def __init__(self, emoji_dim: int = 300, model_path: str = 'emoji2vec.bin'):
        """
        Args:
            emoji_dim: Output dimension of emoji embeddings
            model_path: Path to pre-trained emoji2vec binary file
        """
        super().__init__()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"emoji2vec model not found at {model_path}")
            
        self.emoji2vec = gsm.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.projection = nn.Linear(self.emoji2vec.vector_size, emoji_dim)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Convert text sequences containing emojis to fixed-length embeddings.
        
        Args:
            texts: List of strings potentially containing emojis
            
        Returns:
            Tensor of shape (batch_size, seq_len, emoji_dim) containing emoji embeddings
        """
        batch_size = len(texts)
        device = next(self.projection.parameters()).device
        
        # Pre-allocate output tensor - adding sequence length dimension
        emoji_embeddings = torch.zeros(batch_size, 128, self.emoji2vec.vector_size)
        
        for i, text in enumerate(texts):
            # Extract emojis present in vocabulary
            emojis = [c for c in text if c in self.emoji2vec.key_to_index]
            
            if emojis:
                # Get embeddings for all emojis in text
                emoji_vecs = [self.emoji2vec[emoji] for emoji in emojis]
                # Stack emoji vectors along sequence dimension
                emoji_seq = torch.tensor(np.stack(emoji_vecs))
                # Pad or truncate to fixed sequence length
                if len(emoji_seq) > 128:
                    emoji_embeddings[i] = emoji_seq[:128]
                else:
                    emoji_embeddings[i, :len(emoji_seq)] = emoji_seq
                
        # Move to same device as model and project each embedding in the sequence
        emoji_embeddings = emoji_embeddings.to(device)
        emoji_embeddings = self.projection(emoji_embeddings)
        return emoji_embeddings

def preprocess_urdu_text(text):
    """
    Preprocess Urdu text for better tokenization and model performance.
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

class Attention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector

class SarcasmDetector(nn.Module):
    def __init__(self, dropout_rate=0.5, freeze_bert=False, model_name='xlm-roberta-base'):
        """
        Args:
            model_name: Choose from:
                - 'xlm-roberta-base' (recommended for Urdu)
                - 'bert-base-multilingual-cased' 
                - 'microsoft/mdeberta-v3-base' (if available)
        """
        super(SarcasmDetector, self).__init__()
        
        # Multilingual BERT/RoBERTa components
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get actual model dimension (XLM-R and mBERT both use 768)
        self.bert_dim = self.bert.config.hidden_size
        
        # Add layer normalization
        self.bert_norm = nn.LayerNorm(self.bert_dim)
        self.emoji_norm = nn.LayerNorm(300)
        
        # Emoji components
        self.emoji_encoder = EmojiEncoder()
        
        # Channels and sizes
        self.cnn_out_channels = 256
        self.lstm_hidden_size = 256
        self.dense_hidden_size = 256
        
        # BERT pathway layers
        self.bert_conv = nn.Conv1d(
            in_channels=self.bert_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        self.bert_lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.bert_attention = Attention(self.lstm_hidden_size)
        
        # Emoji pathway layers
        self.emoji_conv = nn.Conv1d(
            in_channels=300,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        self.emoji_lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.emoji_attention = Attention(self.lstm_hidden_size)
        
        # Fusion and classification layers
        combined_features_size = (self.lstm_hidden_size * 4)
        self.fusion = nn.Linear(combined_features_size, self.dense_hidden_size)
        
        self.dense1 = nn.Linear(self.dense_hidden_size, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Add batch normalization
        self.bert_bn = nn.BatchNorm1d(self.cnn_out_channels)
        self.emoji_bn = nn.BatchNorm1d(self.cnn_out_channels)
        
        # Add residual connections
        self.bert_residual = nn.Linear(self.bert_dim, self.lstm_hidden_size * 2)
        self.emoji_residual = nn.Linear(300, self.lstm_hidden_size * 2)

    def process_bert_features(self, embeddings):
        cnn_in = embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.bert_conv(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.bert_lstm(lstm_in)
        features = self.bert_attention(lstm_out)
        return features

    def process_emoji_features(self, embeddings):
        cnn_in = embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.emoji_conv(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.emoji_lstm(lstm_in)
        features = self.emoji_attention(lstm_out)
        return features

    def forward(self, input_ids, attention_mask, raw_texts):
        # Process BERT with residual connection
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = self.bert_norm(bert_output.last_hidden_state)
        bert_residual = self.bert_residual(bert_embeddings.mean(dim=1))
        
        # Process emoji with residual connection
        emoji_embeddings = self.emoji_encoder(raw_texts)
        emoji_embeddings = self.emoji_norm(emoji_embeddings)
        emoji_residual = self.emoji_residual(emoji_embeddings.mean(dim=1))
        
        # Main pathways with proper dimensions
        text_features = self.process_bert_features(bert_embeddings)
        emoji_features = self.process_emoji_features(emoji_embeddings)
        
        # Add residual connections
        text_features = text_features + bert_residual
        emoji_features = emoji_features + emoji_residual
        
        # Combine features
        combined_features = torch.cat([text_features, emoji_features], dim=1)
        
        # Continue through dense layers
        x = self.fusion(combined_features)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        return predictions

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    total_steps = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        raw_texts = batch['raw_text']
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, raw_texts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'  Step [{batch_idx + 1}/{total_steps}], Loss: {loss.item():.4f}')
    
    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), epoch_time

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            raw_texts = batch['raw_text']
            
            outputs = model(input_ids, attention_mask, raw_texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            
            if batch_idx < 3:
                print(f"\nBatch {batch_idx}:")
                for i in range(min(5, len(batch_preds))):
                    print(f"Pred: {batch_preds[i]}, True: {batch_labels[i]}")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    print("\nClass distribution:")
    print("True labels:", np.bincount(all_labels))
    print("Predictions:", np.bincount(all_preds))
    
    print(f"\nDetailed metrics:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Add F1 score for minority class (sarcastic)
    f1_sarcastic = f1_score(all_labels, all_preds, average=None)[1]
    print(f"F1 Score (sarcastic class): {f1_sarcastic:.4f}")
    
    return avg_loss, accuracy, f1_sarcastic, precision, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate Urdu Sarcasm Detector')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default='urdu_sarcasm')
    parser.add_argument('--model', type=str, help='Model to use', 
                       choices=['xlm-roberta-base', 'bert-base-multilingual-cased'],
                       default='xlm-roberta-base')
    
    args = parser.parse_args()
    train_path = f'data/{args.dataset}/train.txt'
    test_path = f'data/{args.dataset}/test.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print(f"Using model: {args.model}")
    
    train_loader, val_loader, test_loader, tokenizer, class_weights = prepare_bert_data(
        train_path,
        test_path,
        batch_size=16,  # Reduced for multilingual models
        model_name=args.model  # Pass model name to data preparation
    )
    
    model = SarcasmDetector(dropout_rate=0.5, freeze_bert=False, model_name=args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    # Separate parameters for different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},  # Lower LR for pretrained model
        {'params': model.emoji_encoder.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() 
                   if not n.startswith('bert.') and not n.startswith('emoji_encoder.')], 
         'lr': 1e-3}
    ])
    
    # Focal Loss for handling potential class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2):
            super().__init__()
            self.gamma = gamma
            
        def forward(self, input, target):
            ce_loss = nn.functional.cross_entropy(input, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
            return focal_loss
    
    criterion = FocalLoss(gamma=2)
    
    # Use cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=25,
        eta_min=1e-6
    )
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    model_path = f'urdu_sarcasm_detector_{args.model.replace("/", "_")}.pth'
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
    
    print("Training model...")
    total_train_time = 0
    
    for epoch in range(25):
        print(f'\nEpoch {epoch+1}/25')
        train_loss, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
        total_train_time += epoch_time
        
        # Evaluate on training set
        print("\nEvaluating on training set:")
        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader, criterion, device)
        
        # Evaluate on test set
        print("\nEvaluating on test set:")
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
        
        print(f'\nEpoch Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Epoch Time: {epoch_time:.2f}s')
        print(f'Total Training Time: {total_train_time/60:.2f}m')

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print("Training complete!")

    model.load_state_dict(torch.load(model_path))
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
    print(f'Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')
