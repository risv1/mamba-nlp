# Mamba NLP in PyTorch

This repository contains a basic PyTorch implementation of the [Mamba](https://arxiv.org/abs/2312.00752) model for NLP, inspired by transformer architectures with a focus on integrating a Selective State Space Model (SSM) and MambaBlock design for improved sequence modeling. The model is designed to handle tasks such as text generation, classification, and other NLP applications.

## Components

1. ssm.py: Defines the SelectiveSSM module, which models sequences using state-space transformations with attention mechanisms, offering an alternative to standard transformer attention for certain tasks.
2. block.py: Implements the MambaBlock, which combines the SelectiveSSM module with feedforward layers and residual connections, providing robust processing of sequences.
3. encode.py: Contains the PositionalEncoding class for adding position-based information to token embeddings, critical for sequence understanding.
4. model.py: Defines the overall MambaForNLP model, which incorporates the SelectiveSSM blocks, positional encodings, and the final output layer for generating predictions.

## Usage

To train the Mamba model, simply run the following command:

```bash
python train.py
```

This will train the model on a text corpus, with the option to visualize training loss and analyze prediction probabilities using a sample_text defined in the file, try updating with a larger corpus for better results.

## Citation

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
