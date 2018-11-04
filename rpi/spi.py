# Add top level of git to path
import sys
sys.path.append("../")

import spidev
import time
import numpy as np
from multiprocessing import Queue, Process

# Uses SPI to send feedback vectors and receive audio
class audioSPI:

    def __init__(self):
        self.spi = spidev.SpiDev() # create spi object
        self.spi.open(0, 0) # open spi port 0, device (CS) 0
        
        # Configuration
        self.spi.max_speed_hz = int(15.6E6)
    
        # Launch SPI transmitter in separate thread
        self.audio_queue = Queue(maxsize = 1)
        self.fb_queue = Queue(maxsize = 1)
        thread = Process(target=self.transmitter,
                         args=(self.fb_queue,
                               self.audio_queue),
                         daemon = True)
        thread.start()
    
    # Meant to be run in a separate thread. Collects audio samples
    # until size received and outputs instance to queue.
    def transmitter(self, queuein, queueout):
    
        # How many samples to acquire before returning an instance
        size = queuein.get()
        
        # Run continuously in separate thread
        while True:
            
            # Get feedback vector if there is one
            try:
                payload = queuein.get_nowait()
            except:
                payload = [0xAA] # Send 10101010 between fb vectors
            
            # Gather samples for instance
            instance = list()
            while len(instance) < size*2: # 2 bytes per sample
                
                # Send/receive 1 byte for 1/2 audio sample
                instance += self.spi.xfer2(payload)
                payload = [0xAA]
            
            # Only one instance allowed in queue
            try:
                queueout.put(instance, block=False)
            except:
                pass
        
        
    def __del__(self):
        self.spi.close() # close the port before exit
        
if __name__ == "__main__":
    import data_processing.audio_processing as ap
    
    # Instantiate communicator
    comm = audioSPI()
    
    # Choose instance size to return
    sample_rate = 44100
    instance_samples = int(sample_rate*ap.INSTANCE_SIZE) # INSTANCE_SIZE = seconds
    comm.fb_queue.put(int(instance_samples/2))
    
    # Test sending a vector
    time.sleep(0.5)
    
    vector = [1]+[0]*41
    spi_vector = np.ones(48, dtype=np.int8)
    spi_vector[3:45] = np.asarray(vector, dtype=np.int8)
    spi_vector = np.packbits(spi_vector)
    
    # Get values
    instance = np.asarray( comm.audio_queue.get(), dtype=np.int16 )
    comm.fb_queue.put(spi_vector.tolist())
    instance1 = np.asarray( comm.audio_queue.get(), dtype=np.int16 )
    
    # Get all even/odd numbers up to instance size
    even = np.arange(start=0, stop=instance_samples+100, step=2)
    odd  = np.arange(start=1, stop=instance_samples+100, step=2)
    
    # Convert 2 bytes to 1 audio sample
    instance[even[0:len(instance)/2]] = np.left_shift(instance[even[0:len(instance)/2]], 8)
    instance = np.bitwise_or(instance[even[0:len(instance)/2]],
                             instance[odd[0:len(instance)/2]])
    
    print(set(instance))
    print(set(instance1))
    
    import pdb;pdb.set_trace()