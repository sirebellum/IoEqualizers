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

# Crop dimensions
HEIGHT = 112
WIDTH = 56
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
               NFFT=256, \
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

    def __init__(self, directory, fb=False):

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
        self.nsynth_dir= os.path.join(os.path.dirname(abs_path), directory)
    
        # Prepare data for accessing wav files / annotations
        json_file = os.path.join(self.nsynth_dir, "examples.json")
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        self.filenames = [ filename for filename in annotations ]
        self.wav_dir = os.path.join(self.nsynth_dir, "audio")
        
        # Prepare data for storing wav files / annotations
        self.dataset = {}
        for clmn in self.clmns:
            self.dataset[clmn] = [ annotations[key][clmn] for key in annotations ]
        self.dataset['filename'] = self.filenames # Add filenames
        
        # Access stats
        self.num_instances = len(annotations)
        self.num_accessed = 0 # Increments for every accessed instance
        
        #Multithreading
        self.num_threads = 10
        #Initialize Threads
        self.pool = Pool(processes=self.num_threads)
        
        # If feedback insertion requested
        self.fb_samples = None
        if fb:
            fb_dir = os.path.join(os.path.dirname(abs_path), "feedback")
            feedback_files = glob.glob(fb_dir+"/*.csv")
            fb_data = feedback(feedback_files)
            # Get all feedback samples, duplicate to 20% of dataset, and shuffle
            self.fb_samples = fb_data.returnInstance(99999, unprocessed=True)
            self.fb_samples = self.fb_samples * int(0.2*self.num_instances/len(self.fb_samples))
            self.num_fb = len(self.fb_samples)
            random.shuffle(self.fb_samples)
        
    # Multithreaded return function. Return num instances
    def returnInstance(self, num):
        
        ffts = None
        if self.num_accessed < self.num_instances:
            del ffts; ffts = list()
            upper = self.num_accessed + num # upper access index

            # Get relevant filenames and prepend full path
            filenames = self.filenames[self.num_accessed:upper]
            filenames = [ os.path.join(self.wav_dir, file+".wav") for file in filenames ]
            
            # Slice correct number of labels     
            data = {}
            for clmn in self.clmns:
                data[clmn] = self.dataset[clmn][self.num_accessed:upper]
                
            # Process wavs
            ffts = self.pool.map(convertWav, filenames)
             
            ### Feedback insertion
            if self.fb_samples is not None:
                # Get a number of feedback samples proportional to the batch size
                insertions = int( len(filenames)/self.num_instances * self.num_fb )
                audio = list()
                feedback = list()
                for i in range(0, insertions):
                    try: # Catch fb_sample pop errors
                        # Pair feedback with random entries (overlay and fft later)
                        random_entry = random.randint(0, len(filenames)-1)
                        audio.append( scipy.io.wavfile.read(filenames[random_entry])[::-1] )# Reverse
                        feedback.append( self.fb_samples.pop() )
                        # Delete used instance & copy labels to new instance
                        del ffts[random_entry]
                        del filenames[random_entry]
                        for clmn in self.clmns:
                            data[clmn].append(data[clmn][random_entry])
                            del data[clmn][random_entry]
                            
                    except IndexError:
                        insertions -= 1
                        print("Ran out of fb samples!")
                        
                # Get overlayed audio samples
                input = list( zip(audio, feedback) )
                audio_wfeedback = self.pool.map(insert_feedback, input)
                # Get ffts of audios with feedback
                ffts += self.pool.map(convertWav, audio_wfeedback)
                # Create feedback label entries
                feedback_labels = [0]*(num-insertions) + [1]*insertions
                data['fb'] = feedback_labels
                
            # Add ffts to output
            data['fft'] = ffts
            
            # Break out data dictionary into per-instance list for shuffling
            temp = list()
            for key in data.keys():
                temp.append(data[key])
            temp = list( zip(*temp) )
            
            # Shuffle and reconbine into dictionary
            random.shuffle(temp)
            temp = list( zip(*temp) )
            temp.reverse()
            for key in data.keys():
                data[key] = temp.pop()
            
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


# Class to pull feedback instances from multiple annotation files
class feedback:
    # annotations is a list of csv files that label feedback files
    # self_sample indicates that the dataset will return non-feedback samples
    def __init__ (self, annotations, self_sample=False, testing=False):
        self.testing = testing # For data cleaning
        
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
                    self.dataset["wavfile"].append( os.path.join(self.wav_dir, entry[0]) )
                    self.dataset["beginning"].append( float(entry[1]) )
                    self.dataset["duration"].append( float(entry[2]) )
                
        # Access stats
        self.num_instances = len(self.dataset["wavfile"])
        self.num_accessed = 0 # Increments for every accessed instance
        
        self.instance_size = 0.6 # 0.6 seconds
        
        ### Ensure feedback durations
        delete = list()
        for x in range(0, self.num_instances):
            # Chop up long instances
            if self.dataset["duration"][x] > self.instance_size:
                delete.append(x) # Delete offending instance later
                num_splits = int(self.dataset["duration"][x] / self.instance_size)+1
                for i in range(0, num_splits):
                    self.dataset['wavfile'].append(self.dataset['wavfile'][x])
                    new_beginning = self.dataset['beginning'][x]+i*self.instance_size
                    self.dataset['beginning'].append(new_beginning)
                    self.dataset['duration'].append(self.instance_size)
                    
            # Pad short instances after
            if self.dataset["duration"][x] < self.instance_size:
                self.dataset["duration"][x] = self.instance_size
        
        # Delete chopped up instances
        for j in sorted(delete, reverse=True):
            del self.dataset['wavfile'][j]
            del self.dataset['beginning'][j]
            del self.dataset['duration'][j]
        
        # New access stats
        self.num_instances = len(self.dataset["wavfile"])
        
        
        ### Non-feedback samples
        if self_sample:
        
            # Append feedback labels for all the actual feedback
            self.dataset['fb'] = []
            for x in range(0, self.num_instances):
                self.dataset['fb'].append(1)
        
            # Function for checking overlap between two samples
            def overlap(beg1, dur1, beg2, dur2):
                end1 = beg1 + dur1
                end2 = beg2 + dur2
                
                if beg1 <= end2 and end1 >= beg2:
                    return True
                else: return False
            
            ### Get up to 4 non-feedback instances adjacent to actual feedback
            for x in range(0, self.num_instances):
                nonfb_beg = list()
                # Pull two instances from right before feedback and two from right after
                buff = self.instance_size/2 # So we don't pick up artifacts from poor labeling
                nonfb_beg.append(self.dataset['beginning'][x] - buff - self.instance_size)
                nonfb_beg.append(nonfb_beg[-1] - 0.01 - self.instance_size)
                nonfb_beg.append(self.dataset['beginning'][x] + self.dataset['duration'][x] + buff)
                nonfb_beg.append(nonfb_beg[-1] + 0.01 + self.instance_size)
                
                # Check new instances for overlap with existing instances
                delete = list()
                for j in range(0, len(nonfb_beg)):
                    for i in range(0, len(self.dataset['beginning'])):
                        if overlap(nonfb_beg[j],
                                   self.instance_size,
                                   self.dataset['beginning'][i],
                                   self.dataset['duration'][i]) \
                        and self.dataset['wavfile'][x] == self.dataset['wavfile'][i]:
                            delete.append(j)
                delete = set(delete) # Get rid of multiple overlaps
                for j in sorted(delete, reverse=True):
                    del nonfb_beg[j]
                
                # Create entries for remaining instances
                if len(nonfb_beg) > 0:
                    for beg in nonfb_beg:
                        self.dataset['wavfile'].append(self.dataset['wavfile'][x])
                        self.dataset['beginning'].append(beg)
                        self.dataset['duration'].append(self.instance_size)
                        self.dataset['fb'].append(0)
            # Done with adjacent feedback instancing
            
            ### Gather non-feedback wav files
            wavs = list( set(self.dataset['wavfile']) )
            wavs = [os.path.join(self.wav_dir, wav) for wav in wavs]
            
            # Add spoken/sung songs
            wav_wildcard = self.wav_dir+"/nus-smc-corpus/**/**/*.wav"
            wavs += glob.glob(wav_wildcard, recursive=True)
            
            # Add spoken digits 
            wav_wildcard = self.wav_dir+"/FSDD/*.wav"
            wavs += glob.glob(wav_wildcard)
            
            # Gather wav files into one dictionary instead of re-reading
            self.wav_dict = {} # WAV Filename: sample_rate, audio
            for filename in set(wavs):
                # Check for unsupported bit-depth
                try:
                    self.wav_dict[filename] = list(scipy.io.wavfile.read(filename))
                except ValueError:
                    continue # Skip wav
                    
                # Average to 1 channel
                if self.wav_dict[filename][1].ndim == 2:
                    dtype = self.wav_dict[filename][1].dtype
                    average = \
                        self.wav_dict[filename][1][:, 0]/2 + self.wav_dict[filename][1][:, 1]/2
                    self.wav_dict[filename][1] = \
                        np.array(average, dtype=dtype)
            
            ### Greedily sample areas of wavs beneach volume threshold
            add_instances = sum(self.dataset['fb'])*50 # num_feedbacks
            added = 0
            threshold = 100 # Avg per sample
            
            ### TODO: Process each wav in concurrently (multiprocessing.pool)
            ###      - Shuffle resulting data and select add_instances
            ### TODO: Create "add_instance" function to get rid of all the effin' append/del
            for wav in self.wav_dict:
                sample_rate, signal = self.wav_dict[wav]
                
                samples = len(signal)
                instance_samples = int(self.instance_size*sample_rate)
                
                beg = 0 
                while beg < samples-instance_samples \
                and added < add_instances:
                    
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
                        for x in range(0, len(self.dataset['beginning'])): 
                            if overlap(beg/sample_rate, 
                                       self.instance_size,
                                       self.dataset['beginning'][x],
                                       self.dataset['duration'][x]) \
                            and wav == self.dataset['wavfile'][x]:
                                beg += instance_samples # Move window forward
                                good = False
                                break
                            else: # Nothing overlapped
                                good = True
                                
                    if good:
                        # We did it!
                        self.dataset['wavfile'].append(wav)
                        self.dataset['beginning'].append(beg/sample_rate)
                        self.dataset['duration'].append(self.instance_size)
                        self.dataset['fb'].append(0)
                        added += 1
                        beg += int(instance_samples*2.5) # Jump ahead
            # Done with non-silent sampling
            
            # New access stats
            self.num_instances = len(self.dataset["wavfile"])
        
        ### Mark silent instances
        delete = list()
        threshold = 2500 # FFT amp threshold
        for x in range(0, self.num_instances):
            if self.dataset['fb'][x] == 1:
                wav_path = os.path.join(self.wav_dir, self.dataset["wavfile"][x])
                beg = self.dataset["beginning"][x]
                end = self.dataset["beginning"][x] + self.dataset["duration"][x]
                
                instance_fft = convertWav(wav_path,
                                      crop_beg=beg,
                                      crop_end=end)
                fft_time_samples = len(instance_fft[0])
                total_fft_volume = sum(sum(abs(instance_fft)))
                volume = total_fft_volume/fft_time_samples
                if volume < threshold: delete.append(x)
        # Delete silent instances
        for j in sorted(delete, reverse=True):
            del self.dataset['wavfile'][j]
            del self.dataset['beginning'][j]
            del self.dataset['duration'][j]
            del self.dataset['fb'][j]
            
        # New access stats
        self.num_instances = len(self.dataset["wavfile"])
                    
        ### Create jittered duplicates of feedback
        for x in range(0, self.num_instances):
            if self.dataset['fb'][x] == 1:
                    right_jitter = self.dataset['beginning'][x]+0.25*self.instance_size
                    self.dataset['wavfile'].append(self.dataset['wavfile'][x])
                    self.dataset['beginning'].append(right_jitter)
                    self.dataset['duration'].append(self.instance_size)
                    self.dataset['fb'].append(1)
                    left_jitter = self.dataset['beginning'][x]-0.25*self.instance_size
                    self.dataset['wavfile'].append(self.dataset['wavfile'][x])
                    self.dataset['beginning'].append(left_jitter)
                    self.dataset['duration'].append(self.instance_size)
                    self.dataset['fb'].append(1)
        
        # Final access stats
        self.num_instances = len(self.dataset["wavfile"])
        print( "{} instances of feedback, {} of non-feedback"\
                    .format(sum(self.dataset['fb']), self.num_instances) )
            
        ### Shuffle everything
        # Break out dataset dictionary into per-instance list for shuffling
        temp = list()
        for key in self.dataset.keys():
            temp.append(self.dataset[key])
        temp = list( zip(*temp) )
        
        # Shuffle and reconbine into dictionary
        random.shuffle(temp)
        temp = list( zip(*temp) )
        temp.reverse()
        for key in self.dataset.keys():
            self.dataset[key] = temp.pop()
        
    # Return num instances
    # If unprocessed is True, function returns raw audio data
    def returnInstance(self, num, unprocessed=False):
        
        ffts = None
        if self.num_accessed < self.num_instances:
            del ffts
            upper = self.num_accessed + num # upper access index

            # Get relevant filenames and prepend full path
            filenames = self.dataset['wavfile'][self.num_accessed:upper]
            # Get relevant chunks
            beg = self.dataset['beginning'][self.num_accessed:upper]
            dur = self.dataset['duration'][self.num_accessed:upper]
            
            fb = None # self_sample
            if 'fb' in self.dataset.keys():
                fb = self.dataset['fb'][self.num_accessed:upper]
            
            # Storage dictionary
            data = {}
            
            # Process feedback chunks
            if not unprocessed: # Get ffts
                sample_rates = [self.wav_dict[filename][0] for filename in filenames]
                ffts = [convertWav(self.wav_dict[filenames[x]][1],
                                   sample_rate=sample_rates[x],
                                   crop_beg=beg[x],
                                   crop_end=beg[x]+dur[x]) \
                            for x in range(0, len(filenames))]
                data['fft'] = ffts
            
            else: # Get raw audio
                sample_rates = [self.wav_dict[filename][0] for filename in filenames]
                audios = [convertWav(self.wav_dict[filenames[x]][1],
                                     sample_rate=sample_rates[x],
                                     crop_beg=beg[x],
                                     crop_end=beg[x]+dur[x],
                                     convert=False) \
                            for x in range(0, len(filenames))]
                
                data['audio'] = audios
                # include sample rates since it's important
                data['sample_rate'] = sample_rates
                
                if self.testing: # Return timestampds for data cleaning
                    data['wav'] = filenames
                    data['beg'] = beg

            # Increment
            self.num_accessed += num

            if fb is not None: data['fb'] = fb
            return data

        # If no more instances to access
        else:
            return None
        
def main():
    from PIL import Image
    import matplotlib.pyplot as plt
    '''
    # nsynth dataset
    dir = "nsynth/nsynth-test/"
    dataset_test = nsynth(dir, fb=True)
    print ("Getting nsynth data...")
    batch = dataset_test.returnInstance(100)
    images = [ plotSpectrumBW(image) for image in batch['fft'] ]

    #import ipdb; ipdb.set_trace()
    for x in range(0, len(images)):
      if batch['fb'][x] == 1:
        img = Image.fromarray(images[x], 'L')
        img.show()
        
    '''
    # Feedback data
    feedback_files = glob.glob("feedback/*.csv")
    dataset_fb = feedback(feedback_files, self_sample=True, testing=True)
    unprocessed = False # Return wav or not
    print ("Getting feedback data...")
    feedbacks = dataset_fb.returnInstance(100, unprocessed=unprocessed)
    
    if not unprocessed:
        ffts = [ plotSpectrumBW(fft) for fft in feedbacks['fft'] ]
    else: # Set up audio output
        import wave
        import pyaudio
        #define stream chunk   
        chunk = 1024
        #instantiate PyAudio  
        p = pyaudio.PyAudio()  
        #open stream  
        stream = p.open(format=pyaudio.paInt16,  
                        channels=1,  
                        rate=feedbacks['sample_rate'][0],  
                        output=True)
                    
    # Play/show
    while feedbacks is not None:
        for x in range(0, len(feedbacks[list(feedbacks.keys())[0]])):
            # Play audio
            if unprocessed:
                if feedbacks['fb'][x] == 1:
              
                    # Write to temp wav file
                    scipy.io.wavfile.write('temp.wav', feedbacks['sample_rate'][x], feedbacks['audio'][x])
                    
                    # Printouts per sample
                    #print(feedbacks['wav'][x], feedbacks['beg'][x])
                    time_amplitude = sum(abs(feedbacks['audio'][x]))/len(feedbacks['audio'][x])
                    fft = convertWav(feedbacks['audio'][x],
                                     sample_rate=feedbacks['sample_rate'][x])
                                     
                    if float("-inf") in fft or float("+inf") in fft:
                        print("inf")
                    else:
                        fft_time_samples = len(fft[0])
                        total_fft_volume = sum(sum(abs(fft)))
                        print(int(total_fft_volume/fft_time_samples), # FFT Volume
                              int(time_amplitude))                    # Time Volume
                    
                    f = wave.open('temp.wav',"rb")  
                    # Begin playback
                    data = f.readframes(chunk)
                    while data:
                        stream.write(data)
                        data = f.readframes(chunk)
            
            # Display FFTs
            else:
                if feedbacks['fb'][x] == 0:
                    plt.imshow(ffts[x])
                    plt.draw(); plt.pause(0.25)
        
        # Get next batch
        feedbacks = dataset_fb.returnInstance(100, unprocessed=unprocessed)
        if not unprocessed:
            ffts = [ plotSpectrumBW(fft) for fft in feedbacks['fft'] ]

    #stop stream  
    stream.stop_stream()
    stream.close()
    #close PyAudio  
    p.terminate()

if __name__ == "__main__":
    main()
