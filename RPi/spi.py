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
        self.spi.max_speed_hz = 122000
    
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
    def transmitter(self, queuein, queue):
        size = queuein.get()
        while True:
        
            # Get feedback vector if there is one
            try:
                payload = queuein.get_nowait()
            except:
                payload = [0xAA] # Send 10101010 between fb vectors
            
            # Gather samples for instance
            instance = list()
            while len(instance) < size:
                instance += self.spi.xfer2(payload)
                payload = [0xAA]
            
            # Only one instance allowed in queue
            try:
                queue.put(instance, block=False)
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
    
    # Test sending something
    time.sleep(0.5)
    comm.fb_queue.put([0xFF, 0xFF, 0xFF])
    print(set(comm.audio_queue.get()))
    print(set(comm.audio_queue.get()))
    
    import pdb;pdb.set_trace()