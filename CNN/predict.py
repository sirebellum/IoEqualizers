from __future__ import print_function
import numpy as np
import json
import requests
import data_processing.audio_processing as ap
from multiprocessing import Process, Queue

# Server data
server = '192.168.12.3:8501' # Rest API port, grpc is 8500
model_name = 'feedback'
signature_name = 'predict_class'
url = "http://"+server+"/v1/models/"+model_name+":predict"

# Function to create instance json for multiple images
# Input image is of type float32 [0, 1]
def create_json(images, signature_name):

    # Assume size is the same
    height = images[0].shape[0]
    width = images[0].shape[1]
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
    global instance_size # Instance size

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
    frame = 0
    while data:
        stream.write(data)
        # Indicate calculation
        if frame*chunk > instance_size:
            queue.put(True)
            frame = 0
        
        frame += 1
        data = f.readframes(chunk) 


    #stop stream  
    stream.stop_stream()
    stream.close()
    #close PyAudio  
    p.terminate()
    
    # End calculations
    queue.put(False)


if __name__ == "__main__":
    import scipy.io.wavfile
    import pyaudio  
    import wave  
    
    # Signal processing signal
    input = "poetry_slam_ringout.wav"
    sample_rate, signal = scipy.io.wavfile.read(input)
    
    # Get ffts
    instance_size = int(sample_rate*1.15) # 1.15 seconds
    ffts = [ap.convertWav(signal[0+x:x+instance_size], sample_rate=sample_rate) \
                for x in range(0, len(signal), instance_size)]
    ffts.reverse()
    
    # Launch wav reader with indicator queue
    execute_queue = Queue()
    wav = Process(target=wav_player,
                  args=(input,
                        execute_queue),
                  daemon = True)
    wav.start()
    
    # Detect feedback
    while execute_queue.get() == True:
        image = ap.plotSpectrumBW(ffts.pop())
        image = np.asarray(image, dtype=np.float32)
        image *= 1/255.0
        
        # Turn into json request
        json_request = create_json([image], signature_name)
        
        # Get predictions
        output = requests.post(url, data=json_request)
        predictions = output.json()['predictions']
        if predictions[0] == 1: print("Feedback!")
    
