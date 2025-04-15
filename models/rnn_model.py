import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len):
        """
        RNN model for text generation
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden state
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Forward pass for RNN
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc_out(output)

    def prompt(self, tokenizer, prompt_text, max_length=50):
        """
        Generate text from a prompt
        Args:
            tokenizer: Tokenizer instance
            prompt_text: Starting text prompt
            max_length: Maximum length of generated text
        Returns:
            Generated text string
        """
        self.eval()
        with torch.no_grad():
            # Tokenize prompt
            tokens = tokenizer.encode(prompt_text)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze().tolist()
            
            input_ids = torch.tensor([tokens])
            generated = tokens.copy()
            
            # Generate tokens
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token = torch.argmax(outputs[0, -1, :]).item()
                
                # Stop if EOS token or max length
                if next_token == 3 or len(generated) >= self.max_seq_len:
                    break
                    
                generated.append(next_token)
                input_ids = torch.tensor([generated])
            
            return tokenizer.decode(generated)