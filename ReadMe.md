# Text Generation Project

This project implements and compares different neural network architectures for text generation, including RNN, LSTM, and Transformer models.

## Project Structure

## Key Features

- **Models Implemented:**

  - RNN (Recurrent Neural Network)
  - LSTM (Long Short-Term Memory)
  - Transformer

- **Training Features:**

  - AdamW optimizer with weight decay
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping with configurable patience
  - Gradient clipping to prevent exploding gradients
  - Comprehensive metrics tracking

- **Text Processing:**
  - SentencePiece BPE tokenization
  - Vocabulary size: 10000
  - Max sequence length: 512

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- SentencePiece
- NLTK (optional, for BLEU score calculation)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/project-2.git
   cd project-2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Training:**

   ```bash
   python main.py
   ```

2. **Text Generation:**

   ```bash
   python generate.py
   ```

3. **Configuration:**
   Edit `config.py` to modify model architecture and training parameters.

## Model Checkpoints

Trained models are saved in `outputs/models/` with the following naming convention:

- `rnnmodel_best.pth`
- `lstmmodel_best.pth`
- `transformermodel_best.pth`

## Results

Generated text samples are saved in `outputs/generated_text/` and training/validation loss plots are saved in `outputs/plots/`.

## Configuration

Key configuration parameters in `config.py`:

```python
MODEL_CONFIG = {
    'vocab_size': 10000,
    'embedding_dim': 256,
    'hidden_dim': 512,
    'max_seq_len': 512,
    'num_heads': 8,
    'num_layers': 4
}

TRAINING_CONFIG = {
    'batch_size': 128,
    'num_epochs': 30,
    'max_sequences': 5000,
    'learning_rate': 0.0005,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'patience': 3,
    'min_delta': 0.001,
    'scheduler_factor': 0.5,
    'scheduler_patience': 2,
    'scheduler_min_lr': 1e-6
}
```
