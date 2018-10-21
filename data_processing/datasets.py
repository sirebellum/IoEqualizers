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
        
        # internal dataset
        self.dataset = {"wav": [],  # Full path to wav
                        "beg": [],  # Beginning of instance in seconds
                        "dur": [],  # Duration of instance in seconds
                        "fb": [],   # Binary feedback class
                        "freqs": []}# List of frequencies where feedback
        
        # Parse each csv
        for file in annotations:
            with open(file, mode='r') as labelfile:
                for line in labelfile:
                    entry = line.strip("\n").split(",")
                    entry.reverse() # to pop off first items
                    self.addInstance(wav=os.path.join(self.wav_dir, entry.pop()),
                                     beg=float(entry.pop()),
                                     dur=float(entry.pop()),
                                     fb=1,
                                     freqs=list(map(int, entry))) # Remaining items are freqs
        
        # Access stats
        self.update_stats()
        self.num_accessed = 0 # Increments for every accessed instance
        
        # Length of instances in seconds
        self.instance_size = INSTANCE_SIZE
        
        ### Ensure feedback durations
        delete = list()
        for x in range(0, self.num_instances):
            # Chop up long instances
            if self.dataset["dur"][x] > self.instance_size:
                delete.append(x) # Delete offending instance later
                num_splits = int(self.dataset["dur"][x] / self.instance_size)+1
                for i in range(0, num_splits):
                    new_beginning = self.dataset['beg'][x]+i*self.instance_size
                    self.addInstance(wav=self.dataset['wav'][x],
                                     beg=new_beginning,
                                     dur=self.instance_size,
                                     fb=1,
                                     freqs=self.dataset['freqs'][x])
            # Pad short instances
            if self.dataset["dur"][x] < self.instance_size:
                self.dataset["dur"][x] = self.instance_size
        
        # Delete chopped up instances
        self.delInstance(*delete)
        
        # New access stats
        self.update_stats()
        
        ### Non-feedback samples
        if self_sample:
        
            ### Gather non-feedback wav files
            wavs = list( set(self.dataset['wav']) )
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
                    if self.dataset['wav'][x] == wav: # Only for same wav
                        labels.append( [self.dataset['beg'][x], \
                                        self.dataset['dur'][x]] )
                
                # listify for function input
                input = [sample_rate, signal, self.instance_size, threshold, labels, wav]
                inputs.append(input)
            
            # Get instances from all wavs via multiprocessing
            pool = Pool(processes=4)
            instances_unformatted = pool.map(ap.slice_audio, inputs)
            instances = list() # Get rid of per-wav lists
            for instance in instances_unformatted:
                instances += instance
            pool.close()
            pool.join()
            
            # Shuffle and choose instances
            random.shuffle(instances)
            instances = instances[0:add_instances]
            
            # Append instances to dataset
            for instance in instances:
                self.addInstance(wav=instance[2],
                                 beg=instance[0],
                                 dur=instance[1],
                                 fb=0,
                                 freqs=[])
            # Done with non-silent sampling
            
            # New access stats
            self.update_stats()
        
        ### Plot histogram of feedback magnitudes if testing
        if self.testing == True:
            volumes = list()
            for x in range(0, self.num_instances):
                if self.dataset['fb'][x] == 1:
                    wav_path = os.path.join(self.wav_dir, self.dataset["wav"][x])
                    beg = self.dataset["beg"][x]
                    end = self.dataset["beg"][x] + self.dataset["dur"][x]
                    
                    instance_fft = ap.convertWav(self.wav_dict[wav_path][1],
                                              sample_rate=self.wav_dict[wav_path][0],
                                              crop_beg=beg,
                                              crop_end=end)[0] # Only fft
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
            wav_path = os.path.join(self.wav_dir, self.dataset["wav"][x])
            beg = self.dataset["beg"][x]
            end = self.dataset["beg"][x] + self.dataset["dur"][x]
            
            instance_fft = ap.convertWav(self.wav_dict[wav_path][1],
                                      sample_rate=self.wav_dict[wav_path][0],
                                      crop_beg=beg,
                                      crop_end=end)[0] # only fft
            fft_time_samples = len(instance_fft[0])
            total_fft_volume = sum(sum(abs(instance_fft)))
            volume = total_fft_volume/fft_time_samples
            
            if volume < threshold:       delete.append(x)
            elif volume == float("inf"): delete.append(x)
            elif fft_time_samples < WIDTH:  delete.append(x)
        # Delete silent instances
        self.delInstance(*delete)
            
        # New access stats
        self.update_stats()
                    
                    
        ### Create jittered duplicates of feedback
        for x in range(0, self.num_instances):
            if self.dataset['fb'][x] == 1:
                right_jitter = self.dataset['beg'][x]+0.25*self.instance_size
                self.addInstance(wav=self.dataset['wav'][x],
                                 beg=right_jitter,
                                 dur=self.instance_size,
                                 fb=1,
                                 freqs=self.dataset['freqs'][x])
                                 
                left_jitter = self.dataset['beg'][x]-0.25*self.instance_size
                self.addInstance(wav=self.dataset['wav'][x],
                                 beg=left_jitter,
                                 dur=self.instance_size,
                                 fb=1,
                                 freqs=self.dataset['freqs'][x])
        
        # Final access stats
        self.update_stats()
        print( "{} instances of feedback, {} of non-feedback"\
                    .format(sum(self.dataset['fb']), self.num_instances) )
        
        
        ### Shuffle everything
        # Break out dataset dictionary into per-instance list for shuffling
        temp = list()
        for key in self.dataset:
            if len(self.dataset[key]) != 0:
                temp.append(self.dataset[key])
        temp = list( zip(*temp) )
        
        # Shuffle and recombine into dictionary
        random.shuffle(temp)
        temp = list( zip(*temp) )
        temp.reverse()
        for key in self.dataset:
            if len(self.dataset[key]) != 0:
                self.dataset[key] = temp.pop()
    
    
    # Updates num_instances while checking for equal instances in each column
    def update_stats(self):
        
        # Check to make sure numer of instances doesn't change in columns
        # unless 0, since no instance added to that column
        count = 0
        for key in self.dataset:
            if len(self.dataset[key]) != 0:
                if count != len(self.dataset[key]) and count != 0:
                    exit("Uneven instance count " + key + "-1")
                count = len(self.dataset[key])
            
        # Update num_instances if we good
        self.num_instances = count
    
    #   Add instance to internal instance dataset while checking
    # for validity of column titles
    def addInstance(self, **kwargs):
    
        # Make sure keys match dataset
        for key in kwargs:
            if key not in self.dataset.keys():
                exit("Invalid key: " + key)
            else:
                self.dataset[key].append( kwargs[key] )
                
    # Delete variable number of instances based on index
    def delInstance(self, *args):
    
        # Delete instances column by column
        for j in sorted(args, reverse=True):
            for key in self.dataset:
                if len(self.dataset[key]) != 0:
                    del self.dataset[key][j]
        
    # Return num instances
    # If unprocessed is True, function returns raw audio data
    def returnInstance(self, num, unprocessed=False):
        
        ffts = None
        if self.num_accessed < self.num_instances:
            del ffts
            # Storage dictionary
            data = {}
            
            upper = self.num_accessed + num # upper access index

            # Get relevant filenames and prepend full path
            filenames = self.dataset['wav'][self.num_accessed:upper]
            # Get relevant chunks
            beg = self.dataset['beg'][self.num_accessed:upper]
            dur = self.dataset['dur'][self.num_accessed:upper]
            freqs = self.dataset['freqs'][self.num_accessed:upper]
            
            fb = None # self_sample
            if 'fb' in self.dataset.keys():
                fb = self.dataset['fb'][self.num_accessed:upper]
                data['fb'] = fb
            
            # Process feedback chunks
            sample_rates = [self.wav_dict[filename][0] for filename in filenames]
            ffts = [ap.convertWav(self.wav_dict[filenames[x]][1],
                                  sample_rate=sample_rates[x],
                                  crop_beg=beg[x],
                                  crop_end=beg[x]+dur[x]) \
                        for x in range(0, len(filenames))]
            ffts, ref_bins = list( zip(*ffts) ) # Unpack ffts and bins
            
            # Convert to image and crop
            data['fft'] = [ap.plotSpectrumBW(fft) for fft in ffts]
            ref_bins = np.asarray(ref_bins)[:, 0:HEIGHT] # Convert and crop
            
            # Max frequency in fft image freq bins
            data['max'] = [max(bins) for bins in ref_bins]
            data['freqs'] = freqs
            
            # Convert to freq vector
            idxs = ap.freqs_to_idx(freqs, ref_bins)
            data['freqs_vector'] = ap.idx_to_vector(idxs, ref_bins)
            
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
    unprocessed = False # Return wav or not
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
                
                # Calc freq bins
                bins = [feedbacks['max'][x]/(ap.HEIGHT-1) * n for n in range(0, ap.HEIGHT)]
                bins = np.asarray(bins)
                #indices = ap.freq_to_idx(feedbacks['freqs'][x], bins)
                #indices = np.asarray(indices)
                indices = ap.vector_to_idx( # add dim to match what function is expecting
                            [feedbacks['freqs_vector'][x]],
                            np.expand_dims(bins,0))
                # Adjust bins to match image indices
                indices = len(ffts[x]) - indices[0]
                # Draw
                if len(indices) != 0:
                    ffts[x][indices] = 255
                # Display FFTs
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
