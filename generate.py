import torch
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.data_processing import TokenizerWrapper
from utils.evaluation import generate_text
from config import MODEL_CONFIG
import os

def generate_with_model(model, tokenizer, prompt, max_length=50):
    """
    Generate text using model's prompt method
    """
    return model.prompt(tokenizer, prompt, max_length)

def main():
    # Load tokenizer
    if not os.path.exists("data/processed/vocab.model"):
        print("Error: Tokenizer model not found. Please run main.py first to train the models.")
        return
        
    tokenizer = TokenizerWrapper("data/processed/vocab.model")
    
    # Update vocab_size based on tokenizer
    MODEL_CONFIG['vocab_size'] = tokenizer.get_vocab_size()

    # Initialize models
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
        'TRANSFORMER': TransformerModel(
            MODEL_CONFIG['vocab_size'],
            MODEL_CONFIG['embedding_dim'],
            MODEL_CONFIG['hidden_dim'],
            MODEL_CONFIG['max_seq_len'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_layers=MODEL_CONFIG['num_layers']
        )
    }

    print("\nAvailable models: RNN, LSTM, Transformer")
    while True:
        model_choice = input("\nChoose model (or 'quit' to exit): ").upper()
        
        if model_choice == 'QUIT':
            break
            
        if model_choice not in models:
            print("Invalid model choice!")
            continue
        
        try:
            # Load model
            model = models[model_choice]
            file_name = model_choice.lower()
                
            model.load_state_dict(torch.load(f"outputs/models/{file_name}model_best.pth"))
            model.eval()
            
            # Get prompt and generate
            prompt = input("Enter prompt: ")
            result = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=50,
                model_type=model_choice
            )
            
            # Print results
            print("\nGenerated Text:")
            print("-" * 50)
            print(result['text'])
            print(f"Perplexity: {result['perplexity']:.2f}")
            print(f"BLEU Score: {result['bleu_score']:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
