''' Reads and processes audio samples from dataset '''
from __future__ import print_function
import json
import os
import scipy.io.wavfile
import numpy as np
from multiprocessing import Pool
import glob
import random
import audioop

# Global sample attributes
HEIGHT = 112
WIDTH = 56
INSTANCE_SIZE = 0.6
REF_RATE = 44100
# Plot BW spectrum picture. Removes infinities
def plotSpectrumBW(data):
    
    # Convert infinity values to max/min values
    data[data == float('-inf')] = data[data > float('-inf')].min()
    data[data == float('+inf')] = data[data < float('+inf')].max()
    
    # Scale to rgb values
    data = np.interp(data, (data.min(), data.max()), (0, 255))
    
    # Round to uint values
    data = np.round(data).astype(np.uint8)
    
    # Slice to make even square
    data = data[len(data)-HEIGHT:len(data), 0:WIDTH]
    return data


# Plot spectrum picture
def plotSpectrum(data):
    fig, ax = plt.subplots(figsize=(data[:,1].size, data[0].size), dpi=1)
    x, y = np.mgrid[:data[:,1].size, #Number of frames
                       :data[0].size]   #Number of filter banks
    ax.pcolormesh(x, y, data) #create color mesh of audio signal
    
    for item in [fig, ax]: #remove frame
        item.patch.set_visible(False)
    ax.set_axis_off() #remove axes


# Convert fft to filter banks
def computeFB(sample_rate, nfilters, pow_frames, NFFT):
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
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8) # Mean norm
    
    return filter_banks

def convertSampleRate(input, inrate, refrate):

# Convert fb sample rate to match reference's
    audio_converted = audioop.ratecv(
        input,          # input
        input.itemsize, # bit depth (bytes)
        1, inrate,      # channels, inrate
        refrate,        # outrate
        None)           # state..?
    audio_converted = np.frombuffer(audio_converted[0], dtype=np.int16)
    
    return audio_converted
    
# Converts WAV file or audio array into frequency spectrum
# frame_size  : how many ms for fft window
# frame_stride: how many ms for fft window slide increment
# crop_beg    : how many seconds to wait to crop
# crop_end    : how many seconds to crop input file to
# NFFT        : how many frequency bins
# visualize   : to visualize or not to visualize, that is the question
# convert     : allow for raw time data to be returned
def convertWav(input, \
               sample_rate=None, \
               frame_size=0.025, \
               frame_stride=0.01, \
               crop_beg=0, \
               crop_end=None, \
               NFFT=512, \
               visualize=False, \
               convert=True):

    # Enable tuple input
    if type(input) is list:
        sample_rate = input[1]
        input = input[0]
               
    signal = input
    # If filename supplied
    if sample_rate is None:
        sample_rate, signal = scipy.io.wavfile.read(input)

    # Crop
    if crop_end is not None:
        duration = len(signal) / sample_rate
        beg = int(crop_beg * sample_rate) - int(sample_rate*0.01)
        if beg < 0: beg = 0 # Wouldn't want to wrap around the end
        end = int(crop_end * sample_rate) + int(sample_rate*0.01)
        signal = signal[beg:end]
    
    # If an fft conversion isn't requested
    if not convert:
        return signal

    ###Pre-emphasize signal
    #PRE_EMPHASIS = 0.97
    #emphasized_signal = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

    ###Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    ###Window filter
    frames_unfiltered= frames.copy()
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

    ###FFT
    mag_frames_raw = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames_raw = ((1.0 / NFFT) * ((mag_frames_raw) ** 2))  # Power Spectrum
    
    ###dB
    #mag_frames = 20 * np.log10(mag_frames_raw)
    pow_frames = 20 * np.log10(pow_frames_raw)
    
    ###Frequency bins
    bins = np.fft.rfftfreq(NFFT, 1/sample_rate)
    
    if visualize:
        ###Plot
        plotSpectrum(pow_frames)
        #plotSpectrum(mag_frames)
        plotSpectrum(computeFB(sample_rate, 40, pow_frames_raw, NFFT))
        
        plt.show()
    
    #filter_banks = computeFB(sample_rate, 28, pow_frames_raw, NFFT)
    
    # Make spectrogram t x f
    return np.rot90(pow_frames), bins


# Overlays feedback on top of audio, randomly jitters to the right
# Returns audio samples
def insert_feedback(input):

    # break out input
    instance = input[0]
    feedback = input[1]

    # instance/feedback[raw_audio, sample_rate]
    audio = instance[0]
    ref_rate = instance[1]
    
    # Convert fb sample rate to match instances's
    fb_converted = audioop.ratecv(
        feedback[0],          # input
        feedback[0].itemsize, # bit depth (bytes)
        1, feedback[1],       # channels, inrate
        ref_rate,             # outrate
        None)                 # state..?
    fb_converted = list(np.frombuffer(fb_converted[0], dtype=np.int16))
    
    # prepend feedback with random length of silence up to 1 second
    # pad to match audio size
    prepend = random.randint(0, ref_rate*1)
    pad = len(audio) - (len(fb_converted) + prepend)
    fb_converted = [0]*prepend + fb_converted + [0]*pad
    
    # add element-wise
    sample = [int(fb_converted[x]/2+audio[x]/2) for x in range(0, len(audio))]
    
    return [sample, ref_rate]


# Returns slices of audio sample of size seconds
# Ensures no overlap with provided labels
# Ensures threshold for volume
def slice_audio(input):

    # Break out input list
    sample_rate, signal, size, threshold, labels, wav = input
    
    samples = len(signal)
    instance_samples = int(size*sample_rate) # Convert seconds to samples
    
    beg = 0
    instances = list()
    good = True
    while beg < samples-instance_samples:
        
        # Calculate avg volume per sample
        volumes = abs(signal[beg:beg+instance_samples])
        volume = sum(volumes)/len(volumes)
        if volume < threshold: # Check for volume
            ''' Check for volume of fft
            fft = convertWav(signal[beg:beg+instance_samples], sample_rate=sample_rate)
            if float("-inf") in fft or float("+inf") in fft:
                print("inf")
            else:
                fft_time_samples = len(fft[0])
                total_fft_volume = sum(sum(abs(fft)))
                print (int(total_fft_volume/fft_time_samples))
            '''
            beg += instance_samples # Move window forward and try again
            continue
        else: # Check for overlap
            for label in labels:
                if overlap(beg/sample_rate, size, label[0], label[1]):
                    beg += instance_samples # Move window forward
                    good = False
                    break
                else: # Nothing overlapped
                    good = True
                    
        if good:
            # We did it!
            instances.append([beg/sample_rate, size, wav])
            beg += instance_samples # Jump ahead
            
    return instances

# Function for checking overlap between two samples
def overlap(beg1, dur1, beg2, dur2):
    end1 = beg1 + dur1
    end2 = beg2 + dur2
    
    if beg1 <= end2 and end1 >= beg2:
        return True
    else: return False

# Display histogram
def histogram(x, name, nbins=10):
    import matplotlib.pyplot as plt
    # the histogram of the data
    n, bins, patches = plt.hist(x, bins=nbins)

    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Log y scale
    plt.yscale('log', nonposy='clip')
    
    plt.show()
    print(bins)

# Frequencies get converted to closest bin indices in bins.
def freqs_to_idx(freqs, bins):
    assert bins.ndim == 2
    assert len(bins) == len(freqs)

    idxs = list()
    for freq, bin in zip(freqs, bins):
        # Figure out where frequencies fit into bins in order
        idx = np.searchsorted(bin, freq, side="left")
        # Adjust indices if freq is closer to lower index
        idx = idx - (np.abs(freq - bin[idx-1]) < np.abs(freq - bin[idx]))
        idxs.append(idx)
    
    return idxs
    
# Map indices to a binary hot vector within range of freq bins
def idx_to_vector(indices, bins):
    assert bins.ndim == 2
    assert bins.shape[0] == len(indices)

    # Create empty (0) arrays with correct shape
    instances = len(indices)
    num_freqs = bins.shape[1]
    vectors = np.zeros((instances, num_freqs), dtype=np.int64)
    # Set 1s
    for vector, index in zip(vectors, indices):
        vector[index] = 1
    
    return vectors

# Map binary hot vector with freq bins to actual frequencies
def vector_to_idx(vectors, bins):
    assert bins.ndim == 2
    assert bins.shape[0] == len(vectors)
    
    # Create lists with indices for one hots
    indices = list()
    for vector in vectors:
        indices.append( np.asarray([x for x in range(0, len(vector)) if vector[x] == 1]) )
    
    return indices
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Get instances
    dataset = {"wavfile": [],
               "beginning": [],
               "duration": [],
               "frequencies": [],
               "max_freq": [],
    }
    file = 'feedback/poetry_slam_ringout.csv'
    with open(file, mode='r') as labelfile:
        for line in labelfile:
            entry = line.strip("\n").split(",")
            entry.reverse()
            dataset["wavfile"].append(     entry.pop() )
            dataset["beginning"].append(   float(entry.pop()) )
            dataset["duration"].append(    float(entry.pop()) )
            dataset["frequencies"].append( list( map(int, entry) ) ) # Remaining items are freqs
            
    # Obtain ffts and bins
    results = [convertWav("feedback/"+dataset['wavfile'][x],
                          crop_beg=dataset['beginning'][x],
                          crop_end=dataset['beginning'][x]+dataset['duration'][x]) \
                for x in range(0, len(dataset['wavfile']))]
    ffts, ref_bins = list( zip(*results) ) # Unpack into separate lists
    
    # Get fft images and crop bins
    images = [ plotSpectrumBW(fft) for fft in ffts ]
    ref_bins = np.asarray(ref_bins)[:, 0:HEIGHT]
    
    # Get bins from frequencies
    idxs = freqs_to_idx(dataset["frequencies"], ref_bins)
    
    # Get frequency vector
    vectors = idx_to_vector(idxs, ref_bins)
    import ipdb; ipdb.set_trace()
    # Print out ffts with frequencies highlighted
    instances = list( zip(images, idxs) )
    for instance in instances:
        # Adjust bins to match image indices
        indices = np.asarray(instance[1])
        indices = len(instance[0]) - indices
        # Draw
        instance[0][indices] = 255
        plt.imshow(instance[0])
        plt.draw(); plt.pause(0.001); input()