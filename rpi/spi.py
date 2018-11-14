# Add top level of git to path
import sys
sys.path.append("../")

import spidev
import time
import numpy as np
from multiprocessing import Queue, Process

# Uses SPI to send feedback vectors and receive audio
class audioSPI:

    def __init__(self, size):
        self.spi = spidev.SpiDev() # create spi object
        self.spi.open(0, 0) # open spi port 0, device (CS) 0

        # Configuration
        self.spi.max_speed_hz = int(8E5)
        self.spi.cshigh = False
        self.spi.mode = 0b01
        self.BLOCKSIZE = 3780 # number of mesages to send at once

        # Variable set up
        self.size = size # How many samples to acquire before returning an instance

        # Launch SPI transmitter in separate thread
        self.audio_queue = Queue(maxsize = 1)
        self.fb_queue = Queue(maxsize = 1)
        self.thread = Process(target=self._transmitter,
                              args=(self.fb_queue,
                                    self.audio_queue),
                              daemon = True)
        self.thread.start()
                              
    # Meant to be run in a separate thread. Collects audio samples
    def _transmitter(self, queuein, queueout):
    
        # Variable setup
        _block = np.zeros(self.BLOCKSIZE, dtype=np.int16)
        filler = [0xAA] * self.BLOCKSIZE # Send 10101010 between fb vectors
    
        # Run continuously in separate thread
        while True:

            # Get feedback vector if there is one
            try:
                payload = queuein.get_nowait()
                payload = payload * int(self.BLOCKSIZE/len(payload))
            except:
                payload = filler

            # Send/receive for audio sample
            _block[0:self.BLOCKSIZE] = self.spi.xfer(payload)[0:self.BLOCKSIZE]

            # Only one instance allowed in queue
            try:
                queueout.put(_block, block=False)
            except:
                pass

                
    # Collect sample
    def transmit(self, payload):
        
        # Variable setup
        _instance = np.zeros(self.size*2, dtype=np.int16) # 2 bytes per sample
        
        # Transmit until instance is full
        while _instance[-1] == 0:
        
            _range = range(0, self.size*2, self.BLOCKSIZE) # 2 bytes per sample
            
            # Gather samples for instance
            for x in _range:
                self.fb_queue.put(payload)
                _instance[x:x+self.BLOCKSIZE] = self.audio_queue.get()

            # Delete invalid messages
            del_indices = np.where( _instance==0xAA )
            _instance = np.delete(_instance, del_indices)
            break

        return _instance
        

    def __del__(self):
        self.spi.close() # close the port before exit

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import data_processing.audio_processing as ap
    from timeit import default_timer as timer

    # Choose instance size to return
    sample_rate = 44100
    instance_samples = int(sample_rate*ap.INSTANCE_SIZE) # INSTANCE_SIZE = seconds
    size = int(instance_samples/2)
    
    # Instantiate communicator
    comm = audioSPI(size)

    # Prep vector to send
    vector = [1]+[0]*41
    spi_vector = np.ones(48, dtype=np.int8)
    spi_vector[3:45] = np.asarray(vector, dtype=np.int8)
    spi_vector = np.packbits(spi_vector)

    # Get all even/odd numbers up to instance size
    even = np.arange(start=0, stop=instance_samples+100, step=2)
    odd  = np.arange(start=1, stop=instance_samples+100, step=2)

    # Obtain and print samples until interrupted
    try:
        while True:
            beg = timer()

            # Get instance and delete values that equal 0xAA
            instance = comm.transmit(spi_vector.tolist())

            # Convert 2 bytes to 1 audio sample
            samples = int(len(instance)/2)
            instance[even[0:samples]] = \
                    np.left_shift(instance[even[0:samples]], 8)
            instance = np.bitwise_or(instance[even[0:samples]],
                                     instance[odd[0:samples]])

            # Visualize
            #plt.plot(instance); plt.draw(); plt.pause(.0001)

            # Print
            print( timer()-beg, ":", set(instance) , len(instance))

    except KeyboardInterrupt:
        exit()
        
