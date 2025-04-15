# Configuration for both model architecture and training parameters

# Model architecture configuration
MODEL_CONFIG = {
    'vocab_size': 10000,       # Size of the vocabulary
    'embedding_dim': 256,      # Dimension of word embeddings
    'hidden_dim': 512,         # Dimension of hidden layers
    'max_seq_len': 512,        # Maximum sequence length
    'num_heads': 8,            # Number of attention heads in Transformer
    'num_layers': 4            # Number of layers in the model
}

# Training specific configuration
TRAINING_CONFIG = {
    'batch_size': 128,         # Number of samples per batch
    'num_epochs': 30,          # Maximum number of training epochs
    'max_sequences': 5000,     # Maximum number of sequences to use
    'learning_rate': 0.0005,   # Initial learning rate
    'weight_decay': 0.01,      # Weight decay for regularization
    'warmup_steps': 200,       # Number of warmup steps for learning rate
    'patience': 3,             # Early stopping patience
    'min_delta': 0.001,        # Minimum improvement for early stopping
    'scheduler_factor': 0.5,   # Factor for reducing learning rate
    'scheduler_patience': 2,   # Patience for learning rate scheduler
    'scheduler_min_lr': 1e-6   # Minimum learning rate
}