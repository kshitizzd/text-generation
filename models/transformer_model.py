import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len, num_heads=8, num_layers=3):
        """
        Transformer model for text generation
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden state
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))
        
        # Simple dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Forward pass for Transformer
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        B, L = x.size()
        
        # Truncate sequences that exceed max_seq_len
        if L > self.max_seq_len:
            x = x[:, :self.max_seq_len]
            L = self.max_seq_len
        
        # Embeddings + positional encoding
        x = self.embedding(x) + self.pos_embedding[:, :L, :]
        x = self.dropout(x)
        
        # Create simple mask for autoregressive generation
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        mask = mask.to(x.device)
        
        # Forward pass through transformer
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)

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
                # Keep input within max sequence length
                if len(generated) > self.max_seq_len:
                    input_ids = torch.tensor([generated[-self.max_seq_len:]])
                else:
                    input_ids = torch.tensor([generated])
                
                # Get model predictions
                outputs = self(input_ids)
                logits = outputs[0, -1, :] / 0.8  # Simple temperature
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Prevent repetition of same token
                if len(generated) > 2 and next_token == generated[-1] == generated[-2]:
                    # Get next most likely token
                    probs[next_token] = 0
                    next_token = torch.multinomial(probs, 1).item()
                
                # Stop conditions
                if next_token == 3 or len(generated) >= self.max_seq_len:
                    break
                    
                generated.append(next_token)
            
            return tokenizer.decode(generated)