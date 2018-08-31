''' Reads and processes audio samples from dataset '''
from __future__ import print_function
import json
import os
import scipy.io.wavfile
import numpy as np

#Converts WAV file into frequency spectrum
#frame_size  : how many ms for fft window
#frame_stride: how many ms for fft window slide increment
#nfilters    : number of filter banks
#crop        : how many seconds to crop input file to
def convertWav(filename, \
               frame_size=0.025, \
               frame_stride=0.01, \
               nfilters=40, \
               crop=4):

    sample_rate, signal = scipy.io.wavfile.read(filename)
    signal = signal[0:int(crop * sample_rate)] #crop to crop seconds default 4

    ###Pre-emphasize signal
    PRE_EMPHASIS = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

    ###Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    ###Window filter
    frames_unfiltered= frames.copy()
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

    ###FFT
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    ###Filter Banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    ###Mean Normalization
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    return filter_banks


# Class for accessing files in the nsynth dataset
class nsynth:

    def __init__(self, directory):

        # Path to dataset relative to this file
        self.directory = os.path.join(os.path.dirname(__file__), directory)
    
        # Get annotations from json file
        json_file = os.path.join(directory, "examples.json")
        with open(json_file, 'r') as f:
            annotations = json.load(f)
            
        # Basic dataset stats
        self.num_instances = len(annotations)
        
        # Convert Nsynth wavfiles to filter banks
        wav_dir = os.path.join(self.directory, "audio")
        self.filter_banks = [ convertWav(os.path.join(wav_dir, wavfile+".wav")) \
                                for wavfile in annotations ]
        
        
        
        #import ipdb; ipdb.set_trace()
        
        
    # Parse and process nsynth instance labels
    def processNsynth(self, instance):
    
        return None

        
def main():
    # nsynth dataset top directory
    dir = "nsynth/nsynth-test/"
    dataset_test = nsynth(dir)


if __name__ == "__main__":
    main()