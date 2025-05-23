import numpy as np
import os

# --- Configuration ---
input_folder = r"Clustered Frequencies"


# --- Maqam types ---
maqam_types = ['kurd', 'nahawand', 'segah', 'hijaz', 'rast', 'saba', 'hijazkar', 'bayat']

# --- Load frequencies, durations, and maqam types from files ---
frequencies, durations, maqam_type_labels = [], [], []

for filename in os.listdir(input_folder):
    if filename.endswith('.npz'):
        maqam_type = next((m for m in maqam_types if filename.lower().startswith(m)), None)
        if maqam_type:
            try:
                data = np.load(os.path.join(input_folder, filename))
                # Convert numpy arrays to Python native types
                freq_array = data['frequencies'].astype(float).tolist()  # Convert to Python floats
                dur_array = data['durations'].astype(int).tolist()       # Convert to Python ints
                frequencies.append(freq_array)
                durations.append(dur_array)
                maqam_type_labels.append(maqam_type)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Unknown maqam type in file: {filename}")

# --- Flatten all data to extract unique tokens ---
all_frequencies = np.concatenate([np.array(lst) for lst in frequencies])
all_durations = np.concatenate([np.array(lst) for lst in durations])
unique_freqs = np.unique(all_frequencies)
unique_durations = np.unique(all_durations)

# --- Create tokenization dictionaries ---
def create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations):
    token_to_value = {0: "PAD"}
    value_to_token = {"PAD": 0}
    token_to_value["UNK"]=1
    value_to_token[1]="UNK"
    current_index = 2

    # Maqam types
    for m in maqam_types:
        token_to_value[current_index] = m
        value_to_token[m] = current_index
        current_index += 1

    # Frequencies (convert to Python floats)
    for f in unique_freqs:
        py_f = float(f)
        token_to_value[current_index] = py_f
        value_to_token[str(py_f)] = current_index
        current_index += 1

    # Durations (convert to Python ints)
    for d in unique_durations:
        py_d = int(d)
        token_to_value[current_index] = py_d
        value_to_token[str(py_d)] = current_index
        current_index += 1

    # EOS
    token_to_value[current_index] = "EOS"
    value_to_token["EOS"] = current_index
    return token_to_value, value_to_token

token_to_value, value_to_token = create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations)
eos_token = value_to_token["EOS"]

# --- Tokenization utilities ---
def tokenize_combined(sequence, value_to_token):
    tokenized_sequence = []
    for element in sequence:
        # Handle numpy types by converting to native Python types
        if isinstance(element, np.generic):
            element = element.item()
        key = str(element)
        if key in value_to_token:
            tokenized_sequence.append(value_to_token[key])
        else:
            tokenized_sequence.append(value_to_token[1])
    tokenized_sequence.append(value_to_token["EOS"])
    return tokenized_sequence

def detokenize_combined(tokenized_sequence, token_to_value):
    return [token_to_value[int(token)] for token in tokenized_sequence]

# --- Build token sequences ---
def build_token_sequences(type_tokens, freq_tokens, dur_tokens, target_length=None, pad_token=0, eos_token=None):
    assert len(type_tokens) == len(freq_tokens) == len(dur_tokens), "Input lists must match in length"
    token_sequences = []
    for t, freqs, durs in zip(type_tokens, freq_tokens, dur_tokens):
        assert len(freqs) == len(durs), "Frequencies and durations must match in length"
        pair_token_count = 2
        max_pairs = (target_length - 2) // pair_token_count if target_length else None

        i = 0
        while i < len(freqs):
            seq = [t]
            end = i + max_pairs if max_pairs else len(freqs)
            for f, d in zip(freqs[i:end], durs[i:end]):
                # Tokenize each element
                f_token = value_to_token[str(float(f))]
                d_token = value_to_token[str(int(d))]
                seq.extend([f_token, d_token])
            i = end

            if eos_token is not None:
                seq.append(eos_token)
            if target_length and len(seq) < target_length:
                seq += [pad_token] * (target_length - len(seq))
            token_sequences.append(seq)
    return token_sequences

# --- Prepare maqam type tokens ---
type_tokens = [value_to_token[m] for m in maqam_type_labels]

# --- Build tokenized sequences ---
sequences = build_token_sequences(type_tokens, frequencies, durations, target_length=1024, pad_token=0, eos_token=eos_token)


import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        Args:
            data (list of lists): The list of sequences to train the model on.
            sequence_length (int): The length of each input sequence.
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        # Number of samples: each sequence can be used to generate (n - sequence_length) samples
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Create the input and target sequence
        input_sequence = sequence[:-1]  # All except the last element
        target_sequence = sequence[1:]  # All except the first element
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.float)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float)
        
        return input_tensor, target_tensor

# Create the dataset

# Sequence length for autoregressive model
context_size = 128

# Create the dataset
dataset = MyDataset(sequences, context_size)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Iterate through the DataLoader
for input_seq, target_seq in dataloader:
    print("Input Sequence:", input_seq)
    print("Target Sequence:", target_seq)

