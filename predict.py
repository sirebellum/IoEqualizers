import sys
import numpy as np
import json
import requests
import data_processing.audio_processing as ap
from multiprocessing import Process, Queue

#Argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id", help='relative path to model to be loaded.')
parser.add_argument("wav", help='relative path to wav to process. "spi" otherwise')
parser.add_argument("--vis", default='True', help='Plot FFT or not.')
parser.add_argument("--server", default='192.168.12.3', help='Models version number.')
parser.add_argument("--suppress", default='True', help='Models version number.')
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
    from timeit import default_timer as timer
    if args.vis == 'True':
        import cv2
    
    # Overlap ratio between instances
    overlap = 0.5
    
    ### If operating on spi
    if args.wav == 'spi':
        import spi.rpi as spi
        
        # Choose instance size to return
        sample_rate = 44100
        instance_samples = int(sample_rate*ap.INSTANCE_SIZE) # INSTANCE_SIZE = seconds
        size = int(instance_samples*overlap)
        
        # Instantiate communicator
        comm = spi.audioSPI(size)
        
    ### If operating on wav
    else:
        import scipy.io.wavfile
        import struct
    
        # Signal processing signal
        input = args.wav
        sample_rate, signal = scipy.io.wavfile.read(input)
        
        # Number of samples in an instance
        instance_samples = int(sample_rate*ap.INSTANCE_SIZE) # INSTANCE_SIZE = seconds
        size = int(instance_samples*overlap)
        
        # Launch wav reader with indicator queue
        execute_queue = Queue(maxsize = 1)
        wav = Process(target=wav_player,
                      args=(input,
                            execute_queue),
                      daemon = True)
        wav.start()
    
    # First half of instance
    prev_fft = np.asarray( [0]*size )
    
    ### Detect feedback
    fft = 0
    spi_vector = [0xAA]
    while fft is not None:
        
        # SPI
        if args.wav == 'spi':
            this_fft = comm.transmit()
            
        # Wav
        else:
            # Get data chunks from audio source and grow fft sample
            this_fft = list()
            while this_fft is not None and len(this_fft) <= instance_samples*overlap:
                this_fft += struct.unpack('1024h', execute_queue.get())
        
        # Convert to image
        fft = np.asarray(np.concatenate((prev_fft,this_fft)), dtype=np.int16)
        fft = ap.convertWav(fft, sample_rate=sample_rate)[0]
        prev_fft = this_fft[len(this_fft)-size:]
        
        # Volume thresholding
        threshold = 0
        fft_time_samples = len(fft[0])
        total_fft_volume = sum(sum(abs(fft)))
        fft_volume = total_fft_volume/fft_time_samples
        print(fft_volume)
        if fft_volume < threshold or fft_volume == float('inf'):
            continue # Don't even process
        
        # Create fft image and compress into png
        image = ap.plotSpectrumBW(fft)
        png_image = create_png(image, c=0)
        #image = np.asarray(image, dtype=np.float32)
        #image *= 1/255.0
        
        # Turn into json request
        json_request = create_json([png_image], signature_name)
        
        # Option to not send to server
        if args.model_id != 'NULL':

            # Send frame to server
            beg = timer()
            try: output = requests.post(url, data=json_request, timeout=0.5) # Don't hang on one frame
            except: continue
            print(sys.getsizeof(json_request), timer()-beg)
        
        # Get predictions from response
            try:
                predictions = output.json()['predictions']
            except:
                print( output.text )
                exit()

        else: predictions = [[0]]

        # If there was feedback detected
        if max(predictions[0]) == 1:
            print("Feedback!")
            
            vectors = np.asarray(predictions, dtype=np.int8)
            
            # SPI
            if args.wav == 'spi':
                # Format vector for SPI
                spi_vector = np.ones(vectors.shape[1]+6, dtype=np.int8)
                spi_vector[:len(vectors[0])] = vectors[0]
                spi_vector = np.packbits(spi_vector).tolist()
                
                # Send to slave
                if args.suppress == "True":
                    comm.put_payload(spi_vector)
            
            # Visualization
            if args.vis == 'True':               
                # Extrapolate bins
                vectors = ap.vector_resize(vectors,
                                           image.shape[0])
                                           
                # Adjust freq bins to match image indices   
                idxs = np.asarray(ap.vector_to_idx(vectors))
                idxs = image.shape[0] - idxs - 1
                
                # Draw on first couple pixels of freq
                image[idxs, 0:5] = 255
                cv2.imshow('image',image); cv2.waitKey(1)
                
        elif args.vis == 'True':
            cv2.imshow('image',image); cv2.waitKey(1)
