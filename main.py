import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
from torch.optim import AdamW
import torch.nn as nn

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Decoder, EncoderWithClassifier
from utilities import Utilities

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling training iterations
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name.
    """
    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  # don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def compute_perplexity(model, data_loader, eval_iters=100):
    """Compute the perplexity of the model on the data in data_loader."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= eval_iters:
                break
                
            x, y = x.to(device), y.to(device)
            loss, _ = model(x, y)
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return perplexity


def train_classifier():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print(f"Vocabulary size is {tokenizer.vocab_size}")

    # Create datasets
    train_dataset = SpeechesClassificationDataset(tokenizer, 'speechesdataset/train_CLS.tsv')
    test_dataset = SpeechesClassificationDataset(tokenizer, 'speechesdataset/test_CLS.tsv')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Initialize model
    model = EncoderWithClassifier(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    ).to(device)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    best_accuracy = -1
    
    for epoch in range(15):  # 15 epochs as specified in the assignment
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if batch_idx % eval_interval == 0:
                print(f'Epoch {epoch+1}/15 - Batch {batch_idx}/{len(train_loader)} - '
                      f'Loss: {loss.item():.4f}')

        # Calculate epoch metrics
        epoch_loss = total_train_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model(inputs)
                predictions = torch.argmax(logits, dim=1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        
        test_accuracy = test_correct / test_total * 100

        print(f'\nEpoch {epoch+1}/15:')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy: {epoch_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            final_model = model
        
        print('-' * 60)

    # Visualize attention patterns
    utils = Utilities(tokenizer, final_model)
    test_sentence = "My fellow Americans, together we can build a stronger nation."
    utils.sanity_check(test_sentence, block_size, "classifier_attentions")
    print("Sanity check complete.")
    print("Final best test accuracy: {:.2f}%".format(best_accuracy))

def train_language_model():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print(f"Vocabulary size is {tokenizer.vocab_size}")

    # Initialize model
    model = Decoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    ).to(device)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Load training data
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test datasets
    test_files = {
        'obama': 'speechesdataset/test_LM_obama.txt',
        'wbush': 'speechesdataset/test_LM_wbush.txt',
        'hbush': 'speechesdataset/test_LM_hbush.txt'
    }
    
    test_loaders = {}
    for name, file_path in test_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loaders[name] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    model.train()
    
    for iter_num, (xb, yb) in enumerate(train_loader):
        if iter_num >= max_iters:
            break

        # Forward pass
        xb, yb = xb.to(device), yb.to(device)
        loss, _ = model(xb, yb)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            train_perplexity = compute_perplexity(model, train_loader)
            print(f"\nIteration {iter_num}:")
            print(f"Train perplexity: {train_perplexity:.2f}")
            
            # Evaluate on test sets
            for name, loader in test_loaders.items():
                test_perplexity = compute_perplexity(model, loader)
                print(f"{name.capitalize()} test perplexity: {test_perplexity:.2f}")
            print('-' * 60)

    utils = Utilities(tokenizer, model)
    test_sentence = "As I've said, there were patriots who supported this war, and patriots who opposed it."
    utils.sanity_check(test_sentence, block_size, "languageModel_attentions")
    print("Sanity check complete.")

def main():
    parser = argparse.ArgumentParser(description='Run transformer tasks for CSE156 PA2')
    parser.add_argument('part', type=str, choices=['part1', 'part2', 'all'], 
                        help='Which part to run: part1 (classifier), part2 (language model), or all')
    
    args = parser.parse_args()
    
    if args.part == 'part1' or args.part == 'all':
        print("Running Part 1: Classification Task")
        train_classifier()
    
    if args.part == 'part2' or args.part == 'all':
        print("\nRunning Part 2: Language Modeling Task")
        train_language_model()

if __name__ == "__main__":
    main()