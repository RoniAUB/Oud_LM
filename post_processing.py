import numpy as np
import sounddevice as sd
CLustered_frequencies=np.load('Clustered_frequencies.npy')
# Parameters
sample_rate = 22050
frequencies = CLustered_frequencies [np.where(CLustered_frequencies<1000)]
total_duration = 180
# 362. # seconds
num_notes = len(frequencies)
duration = total_duration / num_notes





# Sine wave generator

'''
def sine_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

# Combine all sine waves
waveform = np.concatenate([
    sine_wave(freq, duration, sample_rate)
    for freq in frequencies
])

# Normalize waveform
waveform /= np.max(np.abs(waveform))

# Play it
sd.play(waveform, samplerate=sample_rate)
sd.wait()
'''