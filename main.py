import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
import os
from utils import prepare_urdu_bert_data, evaluate

class UrduSarcasmDetector(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased', dropout_rate=0.3, freeze_bert=True):
        super(UrduSarcasmDetector, self).__init__()
        
        # Use multilingual BERT for Urdu support
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT dimension (multilingual BERT is also 768)
        self.bert_dim = self.bert.config.hidden_size
        
        # Architecture parameters
        self.cnn_out_channels = 256
        self.lstm_hidden_size = 128
        self.dense_hidden_size = 64
        
        # CNN layer
        self.conv1d = nn.Conv1d(
            in_channels=self.bert_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Dense layers
        self.dense1 = nn.Linear(self.lstm_hidden_size * 2, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        # Regularization and activation layers
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # BERT embedding layer (frozen)
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state
        
        # CNN feature extraction
        cnn_in = bert_embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.conv1d(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # BiLSTM sequence learning
        lstm_out, _ = self.lstm(lstm_in)
        final_hidden = lstm_out[:, -1, :]
        
        # Classification layers
        x = self.dense1(final_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        return predictions

# Original English model (keep for comparison)
class SarcasmDetector(nn.Module):
    def __init__(self, dropout_rate=0.3, freeze_bert=True):
        super(SarcasmDetector, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = 768
        
        self.cnn_out_channels = 256
        self.lstm_hidden_size = 128
        self.dense_hidden_size = 64
        
        self.conv1d = nn.Conv1d(
            in_channels=self.bert_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.dense1 = nn.Linear(self.lstm_hidden_size * 2, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state
        
        cnn_in = bert_embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.conv1d(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(lstm_in)
        final_hidden = lstm_out[:, -1, :]
        
        x = self.dense1(final_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        return predictions

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # FOR URDU DATASET
    print("=== URDU SARCASM DETECTION ===")
    train_loader, test_loader, tokenizer = prepare_urdu_bert_data(
        '/kaggle/working/Urdu_Sarcasm/Data/Augmented_Data/augmented_data_5k_processed_train.tsv',
        '/kaggle/working/Urdu_Sarcasm/Data/Augmented_Data/augmented_data_5k_processed_test.tsv',
        batch_size=16
    )
    
    # Model parameters for Urdu
    urdu_model_params = {
        'model_name': 'bert-base-multilingual-cased',  # or 'xlm-roberta-base'
        'dropout_rate': 0.3,
        'freeze_bert': True
    }
    
    training_params = {
        'learning_rate': 2e-5,
        'num_epochs': 25,
        'batch_size': 16
    }
    
    urdu_model_path = 'urdu_sarcasm_detector_model.pth'
    
    urdu_model = UrduSarcasmDetector(**urdu_model_params).to(device)
    
    optimizer = torch.optim.AdamW(urdu_model.parameters(), lr=training_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    if os.path.exists(urdu_model_path):
        print(f"Loading Urdu model from {urdu_model_path}")
        urdu_model.load_state_dict(torch.load(urdu_model_path))
        urdu_model.to(device)
        
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(urdu_model, test_loader, criterion, device)
        print(f'Urdu Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        print(f'F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')
    
    print("Training Urdu model...")
    for epoch in range(training_params['num_epochs']):
        loss = train_epoch(urdu_model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Loss: {loss:.4f}')
        if epoch % 5 == 0:
            test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(urdu_model, test_loader, criterion, device)
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
    print("Urdu training complete!")
    torch.save(urdu_model.state_dict(), urdu_model_path)
    print(f"Urdu model saved to {urdu_model_path}")
    
    # Final evaluation
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(urdu_model, test_loader, criterion, device)
    print(f'Final Urdu Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
