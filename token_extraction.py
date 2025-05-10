import numpy as np

# Define a dummy set of frequencies and durations
num_frequencies = 40  # Number of frequencies
dummy_frequencies = [440 + i * 10 for i in range(num_frequencies)]  # Example frequencies (40 frequencies)

num_durations = 40  # Number of durations (equal to the number of frequencies)
dummy_durations = [0.1 + i * 0.1 for i in range(num_durations)]  # Example durations (40 durations)

# Define special tokens
SPECIAL_TOKENS = ["<START>", "<END>", "<PAD>"]

# Define your vocabulary size (this is the number of discrete time intervals)
VOCAB_SIZE = 10  # Example: 10 discrete time intervals

# Define the function to discretize time and tokenize the dataset
def discretize_time(duration, vocab_size, min_duration=0.1, max_duration=1.0):
    """
    Discretizes the continuous duration based on the vocabulary size
    and returns a token for the given duration.
    """
    # Normalize the duration to the range [0, 1]
    normalized_duration = (duration - min_duration) / (max_duration - min_duration)

    # Map the normalized value to the closest bin in the vocab range
    discretized_bin = int(normalized_duration * (vocab_size - 1))

    # Return the corresponding token in the vocabulary
    return f"DUR_{discretized_bin}"

# Define the function to tokenize the dataset
def tokenize_data(frequencies, durations, vocab_size):
    tokenized_data = []
    for freq, dur in zip(frequencies, durations):
        # Frequency token (convert frequency to string)
        freq_token = f"FREQ_{freq}"

        # Discretize the duration and map it to a token
        dur_token = discretize_time(dur, vocab_size)

        # Special token for starting and ending
        start_token = SPECIAL_TOKENS[0]
        end_token = SPECIAL_TOKENS[1]

        # Type token (For this example, we can use "<NOTE>" for each frequency)
        type_token = "<NOTE>"

        # Append the tokens to the list
        tokenized_data.append([start_token, freq_token, dur_token, type_token, end_token])

    return tokenized_data

# Tokenize the dataset with discretized durations
tokenized_dataset = tokenize_data(dummy_frequencies, dummy_durations, VOCAB_SIZE)

# Save the tokenized dataset to a file
file_path = 'dummy_tokenized_dataset.npy'
np.save(file_path, tokenized_dataset)

# Display the tokenized dataset
for tokens in tokenized_dataset:
    print(tokens)

print(f"Dummy dataset saved to: {file_path}")
print(tokenized_dataset[0])