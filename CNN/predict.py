# Add top level of git to path
import sys
sys.path.append("../")

import numpy as np
import json
import requests
import data_processing.audio_processing as ap
from multiprocessing import Process, Queue

#Argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id", help='relative path to model to be loaded.')
parser.add_argument("--server", default='192.168.12.3', help='Models version number.')
args = parser.parse_args()

# Server data
server = args.server+':8501' # Rest API port, grpc is 8500
model_name = args.model_id
signature_name = 'predict_class'
url = "http://"+server+"/v1/models/"+model_name+":predict"

# Function to compress np image into png
# returns base64 encoded string
def create_png(image, c=0):
    import png
    import io
    import base64
    
    # Write to file in memory
    png_image = io.BytesIO()
    png.from_array(image, 'L', info={'compression': c}).save(png_image)
    png_image.seek(0)
    
    # Base64 encode for transmission
    # Alternate characters required by tf
    png_image = base64.b64encode(
                    png_image.read(),
                    altchars=b"-_").decode("utf-8") 
    
    return png_image

# Function to create instance json for multiple images
# Input image is of type float32 [0, 1]
def create_json(images, signature_name):

    # Assume size is the same
    height = ap.HEIGHT
    width = ap.WIDTH
        
    if isinstance(images[0], np.ndarray):
        # Flatten for transmission
        images = [image.flatten().tolist() for image in images]
    
    # Top level request Json with list of instances
    request_dict = {"instances": [], "signature_name": signature_name}
    
    # fill in the request dictionary with the necessary data
    for image in images:
        request_dict["instances"].append( {"image": image, \
                                           "height": height, \
                                           "width" : width} )
    # Jsonize
    json_request = json.dumps(request_dict)
    
    return json_request
    
# Function ment to play wav in other thread, outputing whenever a certain
#   number of frames are processed
def wav_player(filename, queue):
    import pyaudio  
    import wave

    # Audio playback setup
    f = wave.open(filename,"rb")  
    #define stream chunk   
    chunk = 1024
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),  
                    channels=f.getnchannels(),  
                    rate=f.getframerate(),  
                    output=True)  
    
                    
    # Begin playback
    data = f.readframes(chunk)
    while data:
        stream.write(data)
        try:
            queue.put(data, block=False)
        except: pass
        data = f.readframes(chunk) 


    #stop stream  
    stream.stop_stream()
    stream.close()
    #close PyAudio  
    p.terminate()
    
    # End calculations
    queue.put(None)


if __name__ == "__main__":
    import scipy.io.wavfile
    from PIL import Image
    import matplotlib.pyplot as plt
    import struct
    from timeit import default_timer as timer
    
    # Signal processing signal
    input = "test_feedback.wav"
    sample_rate, signal = scipy.io.wavfile.read(input)
    
    # Get ffts
    instance_samples = int(sample_rate*ap.INSTANCE_SIZE) # INSTANCE_SIZE = seconds
    
    # Launch wav reader with indicator queue
    execute_queue = Queue(maxsize = 1)
    wav = Process(target=wav_player,
                  args=(input,
                        execute_queue),
                  daemon = True)
    wav.start()
    
    # Detect feedback
    fft = 0
    counter = 0
    while fft is not None:
        counter += 1
        
        # Get data chunks from wav player and grow fft sample
        fft = list()
        while fft is not None and len(fft) <= instance_samples:
            fft += struct.unpack('1024h', execute_queue.get())
        
        fft = np.asarray(fft, dtype=np.int16)
        fft = ap.convertWav(fft, sample_rate=sample_rate)[0]
        
        # Volume thresholding
        threshold = 2000
        fft_time_samples = len(fft[0])
        total_fft_volume = sum(sum(abs(fft)))
        fft_volume = total_fft_volume/fft_time_samples
        if fft_volume < threshold or fft_volume == float('inf'):
            continue # Don't even process
        
        # Create fft image and compress into png
        image = ap.plotSpectrumBW(fft)
        png_image = create_png(image, c=0)
        image = np.asarray(image, dtype=np.float32)
        image *= 1/255.0
        
        # Turn into json request
        json_request = create_json([png_image], signature_name)
        
        # Get predictions
        beg = timer()
        output = requests.post(url, data=json_request)
        print(sys.getsizeof(json_request), timer()-beg)
        try:
            predictions = output.json()['predictions']
        except:
            print( output.text )
            exit()
        
        if max(predictions[0]) == 1:
            print("Feedback!")
            
            # Extrapolate bins
            vectors = ap.vector_resize(np.asarray(predictions),
                                       image.shape[0])
                                       
            # Adjust freq bins to match image indices   
            idxs = np.asarray(ap.vector_to_idx(vectors))
            idxs = image.shape[0] - idxs - 1
            
            # Draw on first couple pixels of freq
            image[idxs, 0:5] = 1
            plt.imshow(image); plt.draw(); plt.pause(.001)
        else:
            #if counter%10 == 0: print("...")
            plt.imshow(image); plt.draw(); plt.pause(.001)
    
