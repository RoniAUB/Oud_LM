import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyperparameters
VOCAB_SIZE = 512  # Your actual vocab size (use 512 or whatever is correct)
EMBEDDING_DIM = 128
NUM_BLOCKS = 4
NUM_HEADS = 8
TEMPERATURE = 0.7
CONTEXT_SIZE = 5  # Context size = 5 tokens (adjusted to match dataset size)
NUM_EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate

# Load the tokenized dataset
file_path = 'dummy_tokenized_dataset.npy'
tokenized_dataset = np.load(file_path, allow_pickle=True)

# Create token_to_idx dynamically based on the tokenized dataset
all_tokens = [token for seq in tokenized_dataset for token in seq]
token_to_idx = {token: idx for idx, token in enumerate(sorted(set(all_tokens)))}

# Convert tokenized dataset to token indices and pad sequences if necessary
max_sequence_length = 5  # Number of tokens: <START>, FREQ_* , DUR_*, <NOTE>, <END>
indexed_dataset = []
for seq in tokenized_dataset:
    indexed_seq = [token_to_idx[token] for token in seq if token in token_to_idx]

    # If the sequence is shorter than max_sequence_length, pad it
    while len(indexed_seq) < max_sequence_length:
        indexed_seq.append(token_to_idx['<PAD>'])  # Add padding token

    indexed_dataset.append(indexed_seq)

# Convert to tensor
indexed_dataset = torch.tensor(indexed_dataset)


# Transformer model class
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_blocks, num_heads, temperature, context_size):
        super(TransformerModel, self).__init__()

        self.context_size = context_size

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, context_size, embedding_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
            for _ in range(num_blocks)
        ])

        # Final output layer
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        self.temperature = temperature  # Temperature for softmax scaling

    def forward(self, x):
        # Add embeddings and positional encoding
        seq_len = x.size(1)  # Get the actual sequence length from the input
        x = self.embedding(x) + self.positional_encoding[:, :seq_len,
                                :]  # Slice positional encoding to match input size

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final output layer
        logits = self.fc_out(x)
        return logits

    def predict(self, x, prev_token_type=None):
        """Ensure that after a frequency, the next token is duration, and vice versa."""

        logits = self.forward(x)
        out_probs = F.softmax(logits[:, -1, :] / self.temperature, dim=-1)  # Apply temperature scaling

        # Mask out invalid predictions (e.g., after FREQ, don't predict another FREQ)
        freq_tokens = [token for token in token_to_idx if 'FREQ' in token]  # Get all frequency tokens
        dur_tokens = [token for token in token_to_idx if 'DUR' in token]  # Get all duration tokens

        if prev_token_type == 'FREQ':
            out_probs[:, [token_to_idx[token] for token in freq_tokens]] = 0  # Mask FREQ tokens
        elif prev_token_type == 'DUR':
            out_probs[:, [token_to_idx[token] for token in dur_tokens]] = 0  # Mask DUR tokens

        # Normalize the probabilities after masking
        out_probs = out_probs / out_probs.sum(dim=-1, keepdim=True)

        # Sample from the probability distribution
        prediction = torch.multinomial(out_probs, 1)  # Sample one token
        return prediction


# Initialize the model
model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    temperature=TEMPERATURE,
    context_size=CONTEXT_SIZE
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0  # Initialize epoch_loss

    # Zero the gradients
    optimizer.zero_grad()

    # Ensure we don't divide by zero
    num_batches = len(indexed_dataset) // CONTEXT_SIZE
    if num_batches == 0:
        print("Warning: Dataset is too small for the given CONTEXT_SIZE. Skipping training.")
        break

    # Iterate through the dataset in batches
    for batch in range(0, len(indexed_dataset), CONTEXT_SIZE):
        batch_data = indexed_dataset[batch:batch + CONTEXT_SIZE]

        # Ensure batch is the correct size
        if batch_data.size(0) < CONTEXT_SIZE:
            continue

        # Get the input (previous tokens) and the target (next token)
        inputs = batch_data[:, :-1]
        targets = batch_data[:, 1:]

        # Forward pass: Get model output
        output = model(inputs)

        # Calculate loss
        loss = criterion(output.reshape(-1, VOCAB_SIZE), targets.reshape(-1))  # Use reshape instead of view
        epoch_loss += loss.item()  # Accumulate loss for the epoch

        # Backward pass: compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}")

# Example usage: Predicting the next token after training
input_seq = indexed_dataset[0].unsqueeze(0)  # Example input (first sequence in the dataset)
prev_token_type = 'FREQ'  # Previous token type (can be 'FREQ' or 'DUR')

# Get the model's prediction
prediction = model.predict(input_seq, prev_token_type)

# Convert the prediction index back to the token
predicted_token = list(token_to_idx.keys())[list(token_to_idx.values()).index(prediction.item())]
print(f"Predicted next token: {predicted_token}")


