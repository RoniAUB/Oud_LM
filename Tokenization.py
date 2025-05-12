import numpy as np
import os

# --- Load all frequencies from .npy files ---
input_folder =r"\Clustered Frequencies"

frequencies = []
durations=[]
for filename in os.listdir(input_folder):
    if filename.endswith('.npz'):
        try:
            A=np.load(os.path.join(input_folder, filename))
            frequencies.append(A['frequencies'])
            durations.append(A['durations'])
        except Exception as e:
            print(f"Error loading {filename}: {e}")


# Flatten all lists into single arrays
all_frequencies = np.concatenate(frequencies)
all_durations = np.concatenate(durations)

unique_freqs = np.unique(all_frequencies)
unique_durations = np.unique(all_durations)
maqam_types = ['kurd', 'nahawand', 'segah', 'hijaz', 'rast', 'saba', 'hijazkar', 'bayat']

def create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations):
    token_to_value = {0: "PAD"}  # Reserve 0 for padding
    value_to_token = {"PAD": 0}

    current_index = 1  # Start indexing after PAD

    # Maqam types
    for type in maqam_types:
        token_to_value[current_index] = type
        value_to_token[str(type)] = current_index
        current_index += 1

    # Frequencies
    for freq in unique_freqs:
        token_to_value[current_index] = float(freq)
        value_to_token[str(float(freq))] = current_index
        current_index += 1

    # Durations
    for duration in unique_durations:
        token_to_value[current_index] = int(duration)
        value_to_token[str(int(duration))] = current_index
        current_index += 1

    # Add EOS token at the end
    token_to_value[current_index] = "EOS"
    value_to_token["EOS"] = current_index

    return token_to_value, value_to_token

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
    return [token_to_value[token] for token in tokenized_sequence]

def build_token_sequences(type_tokens, freq_tokens, dur_tokens, target_length=None, pad_token=0, eos_token=None):
    assert len(type_tokens) == len(freq_tokens) == len(dur_tokens), "Input lists must have the same length"
    token_sequences = []

    for t, freqs, durs in zip(type_tokens, freq_tokens, dur_tokens):
        assert len(freqs) == len(durs), "Frequency and duration lists must be same length"

        pair_token_count = 2
        max_pairs_per_seq = None
        if target_length is not None:
            assert target_length >= 2 + pair_token_count, "Target length too small"
            max_pairs_per_seq = (target_length - 2) // pair_token_count

        i = 0
        while i < len(freqs):
            seq = [t]
            end = i + max_pairs_per_seq if max_pairs_per_seq else len(freqs)
            for f, d in zip(freqs[i:end], durs[i:end]):
                seq.extend([f, d])
            i = end

            if eos_token is not None:
                seq.append(eos_token)

            if target_length is not None and len(seq) < target_length:
                seq += [pad_token] * (target_length - len(seq))

            token_sequences.append(seq)

    return token_sequences


token_to_value, value_to_token = create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations)
eos_token = value_to_token["EOS"]

# Create the combined dictionaries
token_to_value, value_to_token = create_combined_tokenization_dictionaries(maqam_types, unique_freqs, unique_durations)

# Example sequence
sequence = ['hijaz', 6, 354.36, 2, 310.68, 2]

# Tokenize the sequence
tokenized_sequence = tokenize_combined(sequence, token_to_value, value_to_token)
print("Tokenized sequence:", tokenized_sequence)

# Detokenize the sequence
detokenized_sequence = detokenize_combined(tokenized_sequence, token_to_value)
print("Detokenized sequence:", detokenized_sequence)

def build_token_sequences(type_tokens, freq_tokens, dur_tokens, target_length=None, pad_token=0):
    """
    Builds token sequences from type, frequency, and duration tokens.
    Splits sequences longer than target_length and pads shorter ones.
    
    Args:
        type_tokens (List[int]): List of type tokens (1D).
        freq_tokens (List[List[int]]): List of frequency token lists.
        dur_tokens (List[List[int]]): List of duration token lists.
        target_length (int, optional): Maximum length for each sequence.
        pad_token (int, optional): Padding token value.

    Returns:
        List[List[int]]: List of token sequences.
    """
    assert len(type_tokens) == len(freq_tokens) == len(dur_tokens), "Input lists must have the same length"
    token_sequences = []

    for t, freqs, durs in zip(type_tokens, freq_tokens, dur_tokens):
        assert len(freqs) == len(durs), "Frequency and duration lists must be same length"

        # Number of tokens each (freq, dur) pair takes
        pair_token_count = 2
        max_pairs_per_seq = None
        if target_length is not None:
            assert target_length >= 1 + pair_token_count, "Target length too small to hold even one pair"
            max_pairs_per_seq = (target_length - 1) // pair_token_count

        i = 0
        while i < len(freqs):
            seq = [t]
            end = i + max_pairs_per_seq if max_pairs_per_seq else len(freqs)
            for f, d in zip(freqs[i:end], durs[i:end]):
                seq.extend([f, d])
            i = end

            # Pad if needed
            if target_length is not None and len(seq) < target_length:
                seq += [pad_token] * (target_length - len(seq))

            token_sequences.append(seq)

    return token_sequences

sequences = build_token_sequences(types, frequencies, durations, target_length=7, pad_token=0)

for i, seq in enumerate(sequences):
    print(f"Sequence {i}: {seq}")
    # Tokenize the sequence
    tokenized_sequence = tokenize_combined(seq, token_to_value, value_to_token)
    print("Tokenized sequence:", tokenized_sequence)
