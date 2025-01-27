# Transformer Implementation

This project implements transformer encoders and decoders from scratch for two different NLP tasks:
1. Speech Classification: Predicting which politician gave a speech segment
2. Language Modeling: Predicting the next word in a sequence

## Project Structure

```
.
├── README.md
├── main.py               # Main training script
├── transformer.py        # Transformer model implementations
├── dataset.py           # Dataset classes for both tasks
├── tokenizer.py         # Simple word-level tokenizer
├── utilities.py         # Helper functions for visualization
└── speechesdataset/     # Directory containing the dataset
    ├── train_CLS.tsv    # Training data for classification
    ├── test_CLS.txt     # Test data for classification
    ├── train_LM.txt     # Training data for language modeling
    ├── test_LM_obama.txt    # Test data for Obama speeches
    ├── test_LM_wbush.txt    # Test data for W. Bush speeches
    └── test_LM_hbush.txt    # Test data for H. Bush speeches
```

## Requirements

- Python 3.7+
- PyTorch
- NLTK
- matplotlib

To install the required packages:
```bash
pip install torch nltk matplotlib
```

## Model Architecture

### Encoder (Part 1)
- Transformer encoder with multi-head attention
- Positional embeddings
- Feed-forward classifier on top
- Used for politician speech classification

### Decoder (Part 2)
- GPT-style transformer decoder
- Causal self-attention
- Used for next-word prediction

## Usage

You can run specific parts of the project using command-line arguments:

1. Run only the classification task (Part 1):
```bash
python main.py part1
```

2. Run only the language modeling task (Part 2):
```bash
python main.py part2
```

3. Run both parts:
```bash
python main.py all
```

## Hyperparameters

The following hyperparameters are used for both tasks:
- Batch size: 16
- Block size (max sequence length): 32
- Learning rate: 1e-3
- Embedding dimension: 64
- Number of attention heads: 2
- Number of transformer layers: 4

## Dataset

The dataset consists of speeches from three American politicians:
- Barack Obama (label 0)
- George W. Bush (label 1)
- George H. Bush (label 2)

### Classification Task
- Input: Speech segments
- Output: Politician prediction (0-2)
- Format: Tab-separated files with label and text

### Language Modeling Task
- Input: Text sequences
- Output: Next word prediction
- Format: Plain text files

## Output and Visualization

The training process will:
1. Display training progress and metrics
2. Save the best classification model as 'best_classifier.pt'
3. Generate attention visualization plots
4. Report final accuracies and perplexities

## Implementation Notes

- All transformer components are implemented from scratch
- No use of pre-built transformer libraries
- Custom word-level tokenization using NLTK
- Attention visualization for analysis

## Expected Performance

### Classification Task
- Test accuracy should be in the 80s range
- Training progress logged every `eval_interval` batches

### Language Modeling Task
- Training perplexity: High 100s
- Test perplexity: 300s-400s range for different politicians
- Progress logged every 100 iterations

## Acknowledgments

This project is based on:
1. The original transformer paper by Vaswani et al. (2017)
2. Andrej Karpathy's tutorial on implementing transformer decoders
