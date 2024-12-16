import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MambaForNLP

class SimpleTokenizer:
    def __init__(self):
        # Initialize with special tokens 😵 
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<eos>'}
        self.vocab_size = 3  # Start after special tokens

    def add_word(self, word):
        word = word.lower().strip()
        if word and word not in self.word_to_idx:
            self.word_to_idx[word] = self.vocab_size
            self.idx_to_word[self.vocab_size] = word
            self.vocab_size += 1
    
    def tokenize(self, text):
        words = text.lower().strip().split()
        indices = []
        for word in words:
            if word:
                self.add_word(word) # if the word hasnt been seen before then add it to vocab :D
                indices.append(self.word_to_idx[word])
        indices.append(self.word_to_idx['<eos>'])  # Add EOS token (end of sentence, see the list of special tokens I've mentioned above) 
        return indices
    
    def decode(self, indices):
        return [self.idx_to_word.get(idx.item() if torch.is_tensor(idx) else idx, '<unk>') 
                for idx in indices]

def create_mamba_model(vocab_size=30000):
    model = MambaForNLP(
        vocab_size=vocab_size,
        d_model=256,
        d_state=16,
        d_ff=1024,
        num_layers=6,
        dropout=0.1
    )
    return model

def plot_prediction_probabilities(probs, idx_to_word, k=5):
    plt.figure(figsize=(10, 6))
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    # make sure these are within the vocabulary btw
    valid_indices = [idx.item() for idx in top_k_indices if idx.item() in idx_to_word]
    valid_probs = [prob.item() for prob, idx in zip(top_k_probs, top_k_indices) 
                   if idx.item() in idx_to_word]
    
    if not valid_indices:
        print("No valid predictions found in vocabulary")
        return
    
    plt.bar(range(len(valid_indices)), valid_probs)
    plt.xticks(range(len(valid_indices)), 
               [idx_to_word[idx] for idx in valid_indices], 
               rotation=45)
    plt.title('Top-k Next Word Predictions')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.show()

def main():
    tokenizer = SimpleTokenizer()
    
    # These is sample text, if you want to test the model with your own text, you can replace these with your own corpus.
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Deep learning models process data efficiently"
    ]
    
    for text in sample_texts:
        tokenizer.tokenize(text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model = create_mamba_model(vocab_size=tokenizer.vocab_size)
    model.eval()
    
    sample_text = sample_texts[0]
    print(f"\nProcessing text: {sample_text}")
    
    try:
        token_indices = tokenizer.tokenize(sample_text)
        input_tensor = torch.tensor(token_indices).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0, -1], dim=-1)
        
        print("\nTokenization:")
        print("Original text:", sample_text)
        print("Tokenized indices:", token_indices)
        print("Decoded tokens:", tokenizer.decode(token_indices))
        
        print("\nPlotting top-5 next word predictions...")
        plot_prediction_probabilities(probs, tokenizer.idx_to_word)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        
if __name__ == "__main__":
    main()