''' Reads and processes audio samples from dataset '''
from __future__ import print_function
import json
import os
import scipy.io.wavfile
import numpy as np
from multiprocessing import Pool
import glob
import random
import audio_processing as ap

# Global sample attributes
HEIGHT = ap.HEIGHT
WIDTH = ap.WIDTH
INSTANCE_SIZE = ap.INSTANCE_SIZE
REF_RATE = ap.REF_RATE

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
        
        self.instance_size = INSTANCE_SIZE
        
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
            '''
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
            '''
            ### Gather non-feedback wav files
            wavs = list( set(self.dataset['wavfile']) )
            #wavs = [os.path.join(self.wav_dir, wav) for wav in wavs]
            
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
                        
                # Convert sample rate
                ref_rate = REF_RATE
                if self.wav_dict[filename][0] != ref_rate:
                    self.wav_dict[filename][1] = ap.convertSampleRate(self.wav_dict[filename][1],
                                                                   self.wav_dict[filename][0],
                                                                   ref_rate)
                    self.wav_dict[filename][0] = ref_rate
            
            
            ### Greedily sample areas of wavs above volume threshold
            add_instances = sum(self.dataset['fb'])*50 # num_feedbacks
            threshold = 0 # Avg per sample
            
            # Get inputs for each wav (multiprocessing)
            inputs = list()
            for wav in self.wav_dict:
                sample_rate, signal = self.wav_dict[wav]
                
                # Get overlap labels
                labels = list()
                for x in range(0, self.num_instances):
                    if self.dataset['wavfile'][x] == wav:
                        labels.append( [self.dataset['beginning'][x], \
                                        self.dataset['duration'][x]] )
                
                # listify for function input
                input = [sample_rate, signal, self.instance_size, threshold, labels, wav]
                inputs.append(input)
            
            # Get instances from all wavs via multiprocessing
            pool = Pool(processes=4)
            instances_unformatted = pool.map(ap.slice_audio, inputs)
            instances = list() # 
            for instance in instances_unformatted:
                instances += instance
            pool.close()
            pool.join()
            
            # Shuffle and choose instances
            random.shuffle(instances)
            instances = instances[0:add_instances]
            
            # Append instances to dataset
            for instance in instances:
                self.dataset['wavfile'].append(instance[2])
                self.dataset['beginning'].append(instance[0])
                self.dataset['duration'].append(instance[1])
                self.dataset['fb'].append(0)
            # Done with non-silent sampling
            
            # New access stats
            self.num_instances = len(self.dataset["wavfile"])
        
        ### Plot histogram of feedback magnitudes if testing
        if self.testing == True:
            volumes = list()
            for x in range(0, self.num_instances):
                if self.dataset['fb'][x] == 1:
                    wav_path = os.path.join(self.wav_dir, self.dataset["wavfile"][x])
                    beg = self.dataset["beginning"][x]
                    end = self.dataset["beginning"][x] + self.dataset["duration"][x]
                    
                    instance_fft = ap.convertWav(self.wav_dict[wav_path][1],
                                              sample_rate=self.wav_dict[wav_path][0],
                                              crop_beg=beg,
                                              crop_end=end)
                    fft_time_samples = len(instance_fft[0])
                    total_fft_volume = sum(sum(abs(instance_fft)))
                    volume = total_fft_volume/fft_time_samples
                    if volume == float("inf"): volume = -1
                    volumes.append( volume )
            ap.histogram(volumes, "volumes", nbins=20)
        
        ### Mark silent instances, get rid of short instances
        delete = list()
        threshold = 2000 # FFT amp threshold
        for x in range(0, self.num_instances):
            wav_path = os.path.join(self.wav_dir, self.dataset["wavfile"][x])
            beg = self.dataset["beginning"][x]
            end = self.dataset["beginning"][x] + self.dataset["duration"][x]
            
            instance_fft = ap.convertWav(self.wav_dict[wav_path][1],
                                      sample_rate=self.wav_dict[wav_path][0],
                                      crop_beg=beg,
                                      crop_end=end)
            fft_time_samples = len(instance_fft[0])
            total_fft_volume = sum(sum(abs(instance_fft)))
            volume = total_fft_volume/fft_time_samples
            
            if volume < threshold:       delete.append(x)
            elif volume == float("inf"): delete.append(x)
            elif fft_time_samples < WIDTH:  delete.append(x)
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
            sample_rates = [self.wav_dict[filename][0] for filename in filenames]
            ffts = [ap.convertWav(self.wav_dict[filenames[x]][1],
                               sample_rate=sample_rates[x],
                               crop_beg=beg[x],
                               crop_end=beg[x]+dur[x]) \
                        for x in range(0, len(filenames))]
            data['fft'] = [ap.plotSpectrumBW(fft) for fft in ffts]
            
            if unprocessed: # Get raw audio
                sample_rates = [self.wav_dict[filename][0] for filename in filenames]
                audios = [ap.convertWav(self.wav_dict[filenames[x]][1],
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
    unprocessed = True # Return wav or not
    print ("Getting feedback data...")
    feedbacks = dataset_fb.returnInstance(100, unprocessed=unprocessed)
    
    ffts = feedbacks['fft']
    if unprocessed: # Set up audio output
        from multiprocessing import Queue, Process
        
        # Launch wav reader with indicator queue
        filename_queue = Queue()
        queuedone = Queue()
        wav = Process(target=wav_player,
                      args=[filename_queue,
                            queuedone],
                      daemon = True)
        wav.start()

    # Play/show
    while feedbacks is not None:
        for x in range(0, len(feedbacks[list(feedbacks.keys())[0]])):
            if feedbacks['fb'][x] == 1:
                
                # Display FFTs]
                plt.imshow(ffts[x])
                plt.draw(); plt.pause(0.001)
                
                # Play audio
                if unprocessed:
                    instance = feedbacks['audio'][x]
                    if feedbacks['sample_rate'][x] != REF_RATE:
                        # Convert fb sample rate to match instances's
                        instance = audioop.ratecv(
                            feedbacks['audio'][x],          # input
                            feedbacks['audio'][x].itemsize, # bit depth (bytes)
                            1, feedbacks['sample_rate'][x],       # channels, inrate
                            REF_RATE,             # outrate
                            None)                 # state..?
                        instance = np.frombuffer(instance[0], dtype=np.int16)
                        
                    # Write to temp wav file
                    scipy.io.wavfile.write('temp.wav', REF_RATE, instance)
                    
                    # Printouts per sample
                    #print(feedbacks['wav'][x], feedbacks['beg'][x])
                    time_amplitude = sum(abs(instance))/len(instance)
                    
                    if float("-inf") in ffts[x] or float("+inf") in ffts[x]:
                        print("inf")
                    else:
                        fft_time_samples = len(ffts[x][0])
                        total_fft_volume = sum(sum(abs(ffts[x])))
                        print(int(total_fft_volume/fft_time_samples), # FFT Volume
                              int(time_amplitude))                    # Time Volume
                    
                    # Play audio and wait for finish
                    filename_queue.put("temp.wav")
                    done = queuedone.get()
                    
        
        # Get next batch
        feedbacks = dataset_fb.returnInstance(100, unprocessed=unprocessed)
        ffts = feedbacks['fft']
    
    # End playback
    filename_queue.put(None)

# Function ment to play wav in other thread
def wav_player(queue, queueout):
    import wave
    import pyaudio
    filename = queue.get()
    
    # Audio playback setup
    f = wave.open(filename,"rb")
    
    #define stream chunk   
    chunk = 1024
    silence = chr(0)*chunk*2
    
    # Open pyaudio stream
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),  
                    channels=f.getnchannels(),  
                    rate=f.getframerate(),  
                    output=True)
    
    while filename is not None:
        # Begin playback
        data = f.readframes(chunk)
        while data:
            stream.write(data)
            data = f.readframes(chunk) 
            
        # Indicate finished playing
        queueout.put(1)

        # Get next file
        filename = None
        while filename is None:
            try: filename = queue.get(block=False)
            except:
                stream.write(silence)
                filename = None
        
        f = wave.open(filename,"rb")

    #stop stream  
    stream.stop_stream()
    stream.close()
    #close PyAudio  
    p.terminate()

if __name__ == "__main__":
    main()
