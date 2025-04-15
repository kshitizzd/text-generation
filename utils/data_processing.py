import os
import pickle
import sentencepiece as spm
import json
from glob import glob
import torch

def train_tokenizer_on_all_texts(raw_dir, vocab_size=5000, model_prefix="data/processed/vocab"):
    """
    Train a SentencePiece tokenizer on all text files in a directory
    Args:
        raw_dir: Directory containing raw text files
        vocab_size: Size of the vocabulary
        model_prefix: Prefix for saving the tokenizer model
    Returns:
        TokenizerWrapper instance
    """
    # Combine all text files into a temporary file for training
    temp_file = "data/processed/temp_combined.txt"
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    
    with open(temp_file, 'w', encoding='utf-8') as outfile:
        for txt_file in glob(os.path.join(raw_dir, "*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + "\n")
    
    # Train tokenizer using SentencePiece
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0
    )
    
    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return TokenizerWrapper(f"{model_prefix}.model")

def process_jsonl_data(jsonl_file, output_file, tokenizer):
    """
    Process JSONL format data for evaluation
    Args:
        jsonl_file: Path to input JSONL file
        output_file: Path to output pickle file
        tokenizer: Tokenizer instance
    Returns:
        List of tokenized sequences
    """
    sequence_length = 128
    tokenized_data = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if 'prompt' in entry and 'completion' in entry:
                    full_text = entry['prompt'] + entry['completion']
                    tokens = tokenizer.encode(full_text)
                    
                    if len(tokens) >= 2:
                        if len(tokens) > sequence_length:
                            # Take the first sequence_length tokens
                            tokenized_data.append(tokens[:sequence_length])
                        else:
                            tokenized_data.append(tokens)
        
        print(f"Created {len(tokenized_data)} evaluation sequences from {jsonl_file}")
        
        with open(output_file, "wb") as f:
            pickle.dump(tokenized_data, f)
            
    except Exception as e:
        print(f"Error processing {jsonl_file}: {str(e)}")
    
    return tokenized_data

def process_text_files(raw_dir, output_dir, tokenizer):
    """
    Process text files into tokenized sequences
    Args:
        raw_dir: Directory containing raw text files
        output_dir: Directory to save processed files
        tokenizer: Tokenizer instance
    """
    sequence_length = 128
    
    for txt_file in glob(os.path.join(raw_dir, "*.txt")):
        basename = os.path.basename(txt_file)
        output_file = os.path.join(output_dir, f"{basename[:-4]}_tokenized.pkl")
        
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Get vocabulary size for validation
        vocab_size = tokenizer.get_vocab_size()
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        
        # Filter out tokens that are out of vocabulary range
        tokens = [t for t in tokens if t < vocab_size]
        
        # Create sequences
        sequences = []
        for i in range(0, len(tokens) - sequence_length, sequence_length // 2):
            seq = tokens[i:i + sequence_length]
            if len(seq) == sequence_length:
                sequences.append(seq)
        
        with open(output_file, "wb") as f:
            pickle.dump(sequences, f)

class TokenizerWrapper:
    def __init__(self, model_file):
        """
        Wrapper for SentencePiece tokenizer
        Args:
            model_file: Path to the tokenizer model file
        """
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
    
    def encode(self, text):
        """
        Encode text into token IDs
        Args:
            text: Input text string
        Returns:
            List of token IDs
        """
        return self.sp.encode(text)
    
    def decode(self, tokens):
        """
        Decode token IDs back to text
        Args:
            tokens: List of token IDs or tensor
        Returns:
            Decoded text string
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        return self.sp.decode(tokens)
    
    def get_vocab_size(self):
        """
        Get the size of the vocabulary
        Returns:
            Vocabulary size
        """
        return self.sp.get_piece_size()

if __name__ == "__main__":
    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    print("Starting tokenizer training...")
    try:
        # Train tokenizer on all text files
        tokenizer = train_tokenizer_on_all_texts("data/raw")
        print("Tokenizer training completed successfully")
    
        # Process all individual text files
        process_text_files("data/raw/", "data/processed/", tokenizer)
        
        # Process JSONL files if they exist
        if os.path.exists("data/train.jsonl"):
            print("Processing train.jsonl...")
            process_jsonl_data("data/train.jsonl", "data/processed/train_jsonl_tokenized.pkl", tokenizer)
        if os.path.exists("data/test.jsonl"):
            print("Processing test.jsonl...")
            process_jsonl_data("data/test.jsonl", "data/processed/test_jsonl_tokenized.pkl", tokenizer)
        
        print("All processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")