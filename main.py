import torch
from torch.utils.data import DataLoader
import pickle
import os
import random
from glob import glob
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.data_processing import TokenizerWrapper, train_tokenizer_on_all_texts, process_text_files, process_jsonl_data
from utils.training import train_model, TextDataset, JsonlDataset, load_and_combine_data
from utils.evaluation import generate_text, evaluate_on_jsonl
from config import MODEL_CONFIG, TRAINING_CONFIG

def main():
    # Create directories for processed data and outputs
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/generated_text", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # Initialize tokenizer if it doesn't exist
    if not os.path.exists("data/processed/vocab.model"):
        print("Training tokenizer on raw text files...")
        tokenizer = train_tokenizer_on_all_texts("data/raw", vocab_size=MODEL_CONFIG['vocab_size'])
        process_text_files("data/raw", "data/processed", TokenizerWrapper("data/processed/vocab.model"))
    
    # Load the tokenizer
    tokenizer = TokenizerWrapper("data/processed/vocab.model")
    
    # Update vocab_size based on the actual tokenizer vocabulary
    MODEL_CONFIG['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {MODEL_CONFIG['vocab_size']}")

    # Process JSONL files if not already processed
    jsonl_files = []
    
    train_jsonl = "data/train.jsonl"
    test_jsonl = "data/test.jsonl"
    
    train_jsonl_processed = "data/processed/train_jsonl_tokenized.pkl"
    test_jsonl_processed = "data/processed/test_jsonl_tokenized.pkl"
    
    if os.path.exists(train_jsonl) and not os.path.exists(train_jsonl_processed):
        print("Processing train.jsonl...")
        process_jsonl_data(train_jsonl, train_jsonl_processed, tokenizer)
        jsonl_files.append(train_jsonl_processed)
    elif os.path.exists(train_jsonl_processed):
        jsonl_files.append(train_jsonl_processed)
        
    if os.path.exists(test_jsonl) and not os.path.exists(test_jsonl_processed):
        print("Processing test.jsonl...")
        process_jsonl_data(test_jsonl, test_jsonl_processed, tokenizer)
        jsonl_files.append(test_jsonl_processed)
    elif os.path.exists(test_jsonl_processed):
        jsonl_files.append(test_jsonl_processed)

    # Load and combine data from both sources
    all_data = []
    processed_files = glob("data/processed/*_tokenized.pkl")
    
    # If we have processed files, load and combine them
    if processed_files:
        all_data = load_and_combine_data(
            processed_files,
            [train_jsonl, test_jsonl],
            tokenizer,
            MODEL_CONFIG['max_seq_len'],
            TRAINING_CONFIG['max_sequences']
        )
    
    if not all_data:
        print("No data found or loaded. Please check your data files.")
        return
        
    # Shuffle and split data into training and test sets
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    print(f"Training data size: {len(train_data)} sequences")
    print(f"Test data size: {len(test_data)} sequences")

    # Create data loaders for training and testing
    train_loader = DataLoader(
        TextDataset(train_data, MODEL_CONFIG['max_seq_len']),
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        TextDataset(test_data, MODEL_CONFIG['max_seq_len']),
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Also create specific JSONL dataset loaders for evaluation
    jsonl_eval_loaders = {}
    if os.path.exists(test_jsonl):
        jsonl_eval_loaders['test'] = JsonlDataset(test_jsonl, tokenizer, MODEL_CONFIG['max_seq_len'])
    if os.path.exists(train_jsonl):
        jsonl_eval_loaders['train'] = JsonlDataset(train_jsonl, tokenizer, MODEL_CONFIG['max_seq_len'], max_samples=100)

    # Initialize all models
    models = {
        'RNN': RNNModel(
            MODEL_CONFIG['vocab_size'],
            MODEL_CONFIG['embedding_dim'],
            MODEL_CONFIG['hidden_dim'],
            MODEL_CONFIG['max_seq_len']
        ),
        'LSTM': LSTMModel(
            MODEL_CONFIG['vocab_size'],
            MODEL_CONFIG['embedding_dim'],
            MODEL_CONFIG['hidden_dim'],
            MODEL_CONFIG['max_seq_len']
        ),
        'Transformer': TransformerModel(
            vocab_size=MODEL_CONFIG['vocab_size'],
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            max_seq_len=MODEL_CONFIG['max_seq_len'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_layers=MODEL_CONFIG['num_layers']
        )
    }

    # Train all models
    for model_name, model in models.items():
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        
        # Create optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
                
        # Train the model
        train_model(
            model,
            train_loader, 
            test_loader, 
            num_epochs=TRAINING_CONFIG['num_epochs'],
            optimizer=optimizer,
        )

        # Save the trained model
        torch.save(model.state_dict(), f"outputs/models/{model_name.lower()}_model.pth")

        # Generate sample text to demonstrate model capabilities
        print(f"\nGenerating sample text with {model_name}:")
        prompts = ["Once upon a time", "The story begins"]

        for prompt in prompts:
            result = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=50,
                model_type=model_name
            )
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {result['text']}")
            print(f"Perplexity: {result['perplexity']:.2f}")
            print(f"BLEU Score: {result['bleu_score']:.4f}")
        
        # Evaluate on jsonl test set if available
        if os.path.exists(test_jsonl):
            print(f"\nEvaluating {model_name} on test.jsonl:")
            eval_results = evaluate_on_jsonl(model, tokenizer, test_jsonl, num_samples=50)
            print(f"Samples evaluated: {eval_results['samples_evaluated']}")
            print(f"Accuracy: {eval_results['accuracy']:.4f}")
            print(f"Average perplexity: {eval_results['avg_perplexity']:.2f}")
            print(f"Average BLEU score: {eval_results['avg_bleu']:.4f}")

if __name__ == "__main__":
    # Entry point of the program
    main()