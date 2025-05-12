import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# Hyperparameters
BATCH_SIZE = 64
EMBED_DIM = 256
NHEAD = 8
NUM_LAYERS = 6
DROPOUT = 0.1
LEARNING_RATE = 0.0001
EPOCHS = 50
SEQ_LENGTH = 7  # Should match your sequence length (including EOS/PAD)

# Assuming you have:
# - sequences: your input token sequences (from previous processing)
# - token_to_value: dictionary mapping tokens to values
# - value_to_token: dictionary mapping values to tokens

# Create dataset split (adjust ratios as needed)
train_size = int(0.8 * len(sequences))
val_size = len(sequences) - train_size
train_data, val_data = torch.utils.data.random_split(sequences, [train_size, val_size])


# Dataset and DataLoader
class MusicDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = torch.LongTensor(sequence[:-1])  # All except last token
        y = torch.LongTensor(sequence[1:])  # All except first token
        return x, y


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


train_loader = DataLoader(MusicDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(MusicDataset(val_data), batch_size=BATCH_SIZE, collate_fn=collate_fn)


# Transformer Model
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, DROPOUT)
        self.transformer = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, src, tgt=None):
        src = self.pos_encoder(self.embedding(src))

        if tgt is not None:
            tgt = self.pos_encoder(self.embedding(tgt))
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1))
        else:
            tgt_mask = None

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=self.get_padding_mask(src),
            tgt_key_padding_mask=self.get_padding_mask(tgt) if tgt is not None else None
        )
        return self.fc_out(output)

    def get_padding_mask(self, tensor):
        return (tensor == 0)  # Assuming 0 is padding token


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


# Initialize model
vocab_size = len(token_to_value)
model = MusicTransformer(vocab_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Shift right for autoregressive prediction
        tgt_input = targets[:, :-1]
        tgt_output = targets[:, 1:]

        optimizer.zero_grad()
        output = model(inputs, tgt_input)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1} | Training Loss: {avg_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            tgt_input = targets[:, :-1]
            tgt_output = targets[:, 1:]
            output = model(inputs, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f}')

# Save model
torch.save(model.state_dict(), 'music_transformer.pth')