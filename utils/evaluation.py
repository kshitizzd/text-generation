import torch
import torch.nn.functional as F
import math
import nltk
import os
import numpy as np

# Create a function to ensure NLTK data is downloaded
def ensure_nltk_resources():
    try:
        # Create directory for NLTK data
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
        
        # Verify the download
        from nltk.tokenize import word_tokenize
        word_tokenize("Test sentence")
        return True
    except Exception as e:
        print(f"Warning: NLTK initialization failed. Using basic tokenization instead.")
        return False

# Basic tokenization fallback
def basic_tokenize(text):
    return text.lower().split()

def calculate_perplexity(model, input_text, tokenizer):
    """Simplified perplexity calculation"""
    model.eval()
    with torch.no_grad():
        # Tokenize input
        tokens = tokenizer.encode(input_text)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
        
        # Need at least 2 tokens for perplexity
        if len(tokens) < 2:
            return 100.0  # Return reasonable default for very short sequences
        
        # Prepare input and target
        input_ids = torch.tensor([tokens[:-1]])
        target_ids = torch.tensor([tokens[1:]])
        
        # Get model predictions
        outputs = model(input_ids)
        
        # Calculate loss
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        
        # Calculate perplexity with clamping to avoid extreme values
        perplexity = math.exp(min(loss.item(), 10))  # Clamp to avoid extreme values
        return min(perplexity, 1000.0)  # Cap maximum perplexity

def calculate_bleu(reference_text, generated_text):
    """Simplified BLEU score calculation"""
    # Convert to lowercase and split into words
    ref_words = reference_text.lower().split()
    gen_words = generated_text.lower().split()
    
    # Count matching words
    matches = sum(1 for word in gen_words if word in ref_words)
    
    # Calculate precision
    precision = matches / len(gen_words) if gen_words else 0
    
    return precision

def nucleus_sampling(logits, p=0.9, temperature=0.7):
    """Simplified nucleus sampling"""
    # Apply temperature
    logits = logits / temperature
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff
    cutoff_idx = torch.sum(cumulative_probs < p) + 1
    cutoff_idx = max(1, min(cutoff_idx, len(sorted_probs) - 1))
    
    # Get top probabilities and indices
    top_probs = sorted_probs[:cutoff_idx]
    top_indices = sorted_indices[:cutoff_idx]
    
    # Sample token
    next_token = top_indices[torch.multinomial(top_probs, 1)]
    
    return next_token

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, p=0.9, model_type="transformer"):
    """Simplified text generation with model-specific parameters"""
    model.eval()
    with torch.no_grad():
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
        
        input_ids = torch.tensor([tokens])
        generated = tokens.copy()
        
        # Set model-specific parameters
        if model_type.lower() == "rnn":
            temperature = 0.9
            p = 0.6
            max_length = min(30, max_length)
        elif model_type.lower() == "lstm":
            temperature = 0.8
            p = 0.8
            max_length = min(40, max_length)
        
        # Generate tokens
        for _ in range(max_length):
            # Get model output
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            
            # Sample next token
            next_token = nucleus_sampling(
                next_token_logits,
                p=p,
                temperature=temperature
            ).item()
            
            # Stop if EOS token (assuming it's 3)
            if next_token == 3:
                break
            
            # Add to generated sequence
            generated.append(next_token)
            input_ids = torch.tensor([generated])
            
            # Stop if maximum length reached
            if len(generated) >= model.max_seq_len:
                break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated)
        
        # Calculate metrics
        perplexity = calculate_perplexity(model, generated_text, tokenizer)
        bleu_score = calculate_bleu(prompt, generated_text)
        
        return {
            'text': generated_text,
            'perplexity': perplexity,
            'bleu_score': bleu_score
        }

def calculate_coherence_score(text):
    """
    Calculate a simple coherence score based on sentence structure.
    """
    # Split into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return 0.0
    
    score = 0.0
    for sentence in sentences:
        words = sentence.split()
        
        # Check sentence length (too short or too long sentences reduce score)
        length_score = min(len(words) / 10.0, 1.0) if len(words) < 20 else 20.0 / len(words)
        
        # Check for basic sentence structure (capital letter, ending punctuation)
        structure_score = 0.0
        if sentence and sentence[0].isupper():
            structure_score += 0.5
        if sentence and sentence[-1] in '.!?':
            structure_score += 0.5
            
        score += (length_score + structure_score) / 2

    return score / len(sentences)

# Initialize NLTK resources when the module is imported
ensure_nltk_resources()