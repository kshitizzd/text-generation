import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        """
        Base class for all text generation models
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, x):
        """
        Forward pass must be implemented by subclasses
        Args:
            x: Input tensor
        """
        raise NotImplementedError("Subclasses must implement forward")

    def generate(self, tokenizer, prompt, max_length=50, temperature=0.7, device="cpu"):
        """
        Generate text from a prompt
        Args:
            tokenizer: Tokenizer instance
            prompt: Starting text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            device: Device to use for computation
        """
        self.eval()
        try:
            # Tokenize the prompt
            tokens = tokenizer.encode(prompt)
            
            # Handle tensor output from tokenizer
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]
            
            # Convert to tensor and move to device
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
            generated = tokens.copy() if isinstance(tokens, list) else tokens[:]
            
            # Generate sequence
            with torch.no_grad():
                for _ in range(max_length):
                    # Ensure we don't exceed the model's max sequence length
                    curr_input = input_ids[:, -self.max_seq_len:]
                    
                    # Forward pass
                    outputs = self(curr_input)
                    
                    # Get predictions for next token
                    next_token_logits = outputs[0, -1, :] / temperature
                    
                    next_token_probs = torch.softmax(next_token_logits.cpu(), dim=-1)
                    next_token = torch.multinomial(next_token_probs, 1).item()
                    
                    # Append to generated sequence
                    generated.append(next_token)
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
                    
            # Decode the generated sequence
            return tokenizer.decode(generated)
            
        except Exception as e:
            print(f"Generation error details: {str(e)}")
            return f"Error in generation: {str(e)}"