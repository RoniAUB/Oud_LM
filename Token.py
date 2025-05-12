import numpy as np
import os

# --- Configuration ---
input_folder = r"\Clustered Frequencies"

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
                frequencies.append(data['frequencies'])
                durations.append(data['durations'])
                maqam_type_labels.append(maqam_type)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Unknown maqam type in file: {filename}")

# --- Flatten all data to extract unique tokens ---
all_frequencies = np.concatenate(frequencies)
all_durations = np.concatenate(durations)
unique_freqs = np.unique(all_frequencies)
unique_durations = np.unique(all_durations)

# --- Create tokenization dictionaries ---
def create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations):
    token_to_value = {0: "PAD"}  # Reserve 0 for padding
    value_to_token = {"PAD": 0}
    current_index = 1

    for m in maqam_types:
        token_to_value[current_index] = m
        value_to_token[m] = current_index
        current_index += 1

    for f in unique_freqs:
        token_to_value[current_index] = float(f)
        value_to_token[str(float(f))] = current_index
        current_index += 1

    for d in unique_durations:
        token_to_value[current_index] = int(d)
        value_to_token[str(int(d))] = current_index
        current_index += 1

    token_to_value[current_index] = "EOS"
    value_to_token["EOS"] = current_index
    return token_to_value, value_to_token

token_to_value, value_to_token = create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations)
eos_token = value_to_token["EOS"]

# --- Tokenization utilities ---
def tokenize_combined(sequence, token_to_value, value_to_token):
    tokenized_sequence = []
    for element in sequence:
        key = str(element)
        if key in value_to_token:
            tokenized_sequence.append(value_to_token[key])
        else:
            raise ValueError(f"Unknown element for tokenization: {element}")
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
                seq.extend([f, d])
            i = end

            if eos_token:
                seq.append(eos_token)
            if target_length and len(seq) < target_length:
                seq += [pad_token] * (target_length - len(seq))
            token_sequences.append(seq)
    return token_sequences

# --- Prepare maqam type tokens ---
type_tokens = [value_to_token[m] for m in maqam_type_labels]

# --- Build tokenized sequences ---
sequences = build_token_sequences(type_tokens, frequencies, durations, target_length=7, pad_token=0, eos_token=eos_token)

# --- Show a few sequences ---
for i, seq in enumerate(sequences[:5]):
    print(f"Sequence {i}: {seq}")
    print("Detokenized:", detokenize_combined(seq, token_to_value))
