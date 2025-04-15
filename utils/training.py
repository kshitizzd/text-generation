import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
import pickle


class TextDataset(Dataset):
    def __init__(self, sequences, max_seq_len):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        seq = torch.tensor(seq, dtype=torch.long)
        return seq[:-1], seq[1:]


class JsonlDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_seq_len, max_samples=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        # Load and tokenize data from JSONL file
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        entry = json.loads(line.strip())
                        if 'prompt' in entry and 'completion' in entry:
                            # Tokenize the prompt and completion
                            prompt_tokens = tokenizer.encode(entry['prompt'])
                            completion_tokens = tokenizer.encode(entry['completion'])
                            
                            # Combine the tokens (prompt + completion)
                            combined_tokens = prompt_tokens + completion_tokens
                            
                            # Trim if too long
                            if len(combined_tokens) > max_seq_len:
                                combined_tokens = combined_tokens[:max_seq_len]
                            
                            if len(combined_tokens) >= 2:  # Need at least 2 tokens for input/target
                                self.samples.append(combined_tokens)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading jsonl file {jsonl_file}: {str(e)}")
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        seq = torch.tensor(seq, dtype=torch.long)
        return seq[:-1], seq[1:]  # Input is all but last token, target is all but first token


def train_model(model, train_loader, test_loader, num_epochs, optimizer, config=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'] if config else 0.5,
        patience=config['scheduler_patience'] if config else 2,
        min_lr=config['scheduler_min_lr'] if config else 1e-6,
        verbose=True
    )

    # Early stopping variables
    patience = config['patience'] if config else 3
    min_delta = config['min_delta'] if config else 0.001
    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            batch_count += 1

            # if batch_idx % 10 == 0:
            #     print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = total_val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Save the best model
            model_name = type(model).__name__.lower()
            torch.save(model.state_dict(), f"outputs/models/{model_name}_best.pth")
            print(f"New best model saved!")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break
    # Save the loss plot
    model_name = type(model).__name__.lower()
    save_plot(train_losses, val_losses, model_name)
    print(f"Loss plot saved for {model_name}")
    return {'train_losses': train_losses, 'val_losses': val_losses}


def save_plot(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/plots/{model_name.lower()}_loss.png')
    plt.close()


def load_and_combine_data(tokenized_files, jsonl_files, tokenizer, max_seq_len, max_sequences):
    """
    Load and combine data from tokenized files and JSONL files
    """
    all_data = []
    
    # Load tokenized data (from .pkl files)
    for file_path in tokenized_files:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                all_data.extend(data[:max_sequences // len(tokenized_files)])
        except Exception as e:
            print(f"Error loading tokenized file {file_path}: {str(e)}")
    
    # Load JSONL data
    for jsonl_file in jsonl_files:
        if os.path.exists(jsonl_file):
            jsonl_dataset = JsonlDataset(jsonl_file, tokenizer, max_seq_len, max_samples=max_sequences//len(jsonl_files))
            for i in range(len(jsonl_dataset)):
                seq, _ = jsonl_dataset[i]
                all_data.append(seq.tolist())
    
    return all_data