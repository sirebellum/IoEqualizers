''' Reads and processes audio samples from dataset '''
from __future__ import print_function
import json
import os
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob

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
    #data = data[0:112, 0:112]
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
               crop_end=4, \
               NFFT=256, \
               visualize=False, \
               convert=True):

    signal = input
    # If filename supplied
    if sample_rate is None:
        sample_rate, signal = scipy.io.wavfile.read(input)

    # Crop
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
    
    if visualize:
        ###Plot
        plotSpectrum(pow_frames)
        #plotSpectrum(mag_frames)
        plotSpectrum(computeFB(sample_rate, 40, pow_frames_raw, NFFT))
        
        plt.show()
    
    #filter_banks = computeFB(sample_rate, 28, pow_frames_raw, NFFT)
    
    # Make spectrogram t x f
    return np.rot90(pow_frames)


# Class for accessing files in the nsynth dataset
class nsynth:

    def __init__(self, directory):

        self.clmns= ["note",
            "note_str",
            "instrument",
            "instrument_str",
            "pitch",
            "velocity",
            "sample_rate",
            "qualities",
            "qualities_str",
            "instrument_family",
            "instrument_family_str",
            "instrument_source",
            "instrument_source_str"]
    
        # Path to dataset relative to this file
        abs_path = os.path.abspath(__file__) # Absolute path of this file
        self.directory = os.path.join(os.path.dirname(abs_path), directory)
    
        # Prepare data for accessing wav files / annotations
        json_file = os.path.join(self.directory, "examples.json")
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        self.filenames = [ filename for filename in annotations ]
        self.wav_dir = os.path.join(self.directory, "audio")
        
        # Prepare data for storing wav files / annotations
        self.dataset = {}
        for clmn in self.clmns:
            self.dataset[clmn] = [ annotations[key][clmn] for key in annotations ]
        self.dataset['filename'] = self.filenames # Add filenames
        
        # Access stats
        self.num_instances = len(annotations)
        self.num_accessed = 0 # Increments for every accessed instance
        
        #Multithreading
        self.num_threads = 4
        #Initialize Threads
        self.pool = Pool(processes=self.num_threads)
        
    
    # Multithreaded return function. Return num instances
    def returnInstance(self, num):
        
        ffts = None
        if self.num_accessed < self.num_instances:
            del ffts
            upper = self.num_accessed + num # upper access index

            # Get relevant filenames and prepend full path
            filenames = self.filenames[self.num_accessed:upper]
            filenames = [ os.path.join(self.wav_dir, file+".wav") for file in filenames ]
            # Process wav chunks
            ffts = self.pool.map(convertWav, filenames)

            # Slice correct number of labels     
            data = {}
            for clmn in self.clmns:
                data[clmn] = self.dataset[clmn][self.num_accessed:upper]
            data['fft'] = ffts

            # Increment
            self.num_accessed += num

            return data

        # If no more instances to access
        else:
            # kill threads
            self.pool.close()
            self.pool.join()
                
            return None
        '''
        if self.num_accessed < self.num_instances:
          
            upper = self.num_accessed + num # upper access index
            # Convert Nsynth wavfiles to fft spectrums
            ffts = [ convertWav(os.path.join(self.wav_dir, self.filenames[x]+".wav")) \
                                    for x in range(self.num_accessed, upper) \
                                    if x < self.num_instances ]
            # Slice correct number of labels     
            data = {}
            for clmn in self.clmns:
                data[clmn] = self.dataset[clmn][self.num_accessed:upper]
            data['fft'] = ffts
            
            # Increment
            self.num_accessed += num
        
            return data
            
        else: return None
        '''

# Class to pull feedback instances from multiple annotation files
class feedback:
    # annotations is a list of csv files that label feedback files
    def __init__ (self, annotations):
        
        # Path to dataset relative to this file
        abs_path = os.path.abspath(__file__) # Absolute path of this file
        self.wav_dir = os.path.join(os.path.dirname(abs_path), "feedback")
        
        # internal annotations
        self.dataset = {"wavfile": [], "beginning": [], "duration": []}
        
        # Parse each csv
        for file in annotations:
            with open(file, mode='r') as labelfile:
                for line in labelfile:
                    entry = line.strip("\n").split(",")
                    self.dataset["wavfile"].append( entry[0] )
                    self.dataset["beginning"].append( float(entry[1]) )
                    self.dataset["duration"].append( float(entry[2]) )
                
        # Access stats
        self.num_instances = len(annotations)
        self.num_accessed = 0 # Increments for every accessed instance
        
    # Multithreaded return function. Return num instances
    # If unprocessed is True, function returns raw wav data
    def returnInstance(self, num, unprocessed=False):
        
        ffts = None
        if self.num_accessed < self.num_instances:
            del ffts
            upper = self.num_accessed + num # upper access index

            # Get relevant filenames and prepend full path
            filenames = self.dataset['wavfile'][self.num_accessed:upper]
            filenames = [ os.path.join(self.wav_dir, file) for file in filenames ]
            # Get relevant chunks
            beg = self.dataset['beginning'][self.num_accessed:upper]
            dur = self.dataset['duration'][self.num_accessed:upper]
            # Process feedback chunks
            if not unprocessed:
                ffts = [convertWav(filenames[x], crop_beg=beg[x], crop_end=beg[x]+dur[x]) \
                            for x in range(0, len(filenames))]
            else:
                ffts = [convertWav(filenames[x], crop_beg=beg[x], crop_end=beg[x]+dur[x], convert=False) \
                            for x in range(0, len(filenames))]

            # Increment
            self.num_accessed += num

            return ffts

        # If no more instances to access
        else:
            return None
        
def main():
    from PIL import Image
    # nsynth dataset top directory
    dir = "nsynth/nsynth-test/"
    dataset_test = nsynth(dir)

    batch = dataset_test.returnInstance(100)
    images = [ plotSpectrumBW(image) for image in batch['fft'] ]

    #import ipdb; ipdb.set_trace()
    for x in range(0, len(images)):
      if batch['instrument_family_str'][x] == "none":
        img = Image.fromarray(images[x], 'L')
        img.show()
        
        
    # Feedback data
    feedback_files = glob.glob("feedback/*.csv")
    dataset_fb = feedback(feedback_files)
    unprocessed = False # Return wav or not
    feedbacks = dataset_fb.returnInstance(10, unprocessed=unprocessed)
    if not unprocessed:
        feedbacks = [ plotSpectrumBW(fft) for fft in feedbacks ]
    
    for x in range(0, len(feedbacks)):
        if unprocessed:
            plt.plot(feedbacks[x])
            plt.ylabel('feedback sample')
            plt.show()
        else:
            img = Image.fromarray(feedbacks[x], 'L')
            img.show()

if __name__ == "__main__":
    main()