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

        # SPI Configuration
        self.spi.max_speed_hz = int(1.953E6)
        self.spi.cshigh = False
        self.spi.mode = 0b01

        # Variable setup
        self.BLOCKSIZE = 3780 # number of mesages to send at once
        self.size = size # number of samples to acquire before returning an instance
        self.bytes = size*2 + self.BLOCKSIZE # 2 bytes per sample + extra for filler
        self.blocks = int(self.size/self.BLOCKSIZE) # blocks per instance
        
        # Arrays for data collection
        self._instance = np.zeros(self.bytes, dtype=np.int16)
        self.instance = np.zeros(self.size, dtype=np.int16)
        
        # All even/odd numbers up to bytes for merging samples
        self._even = np.arange(start=0, stop=self.bytes, step=2)
        self._odd  = np.arange(start=1, stop=self.bytes, step=2)
        
        # Launch SPI transmitter in separate thread
        self.audio_queue = Queue(maxsize = self.blocks)
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
        _filler = [0xAA] * self.BLOCKSIZE # Send 10101010 between fb vectors
    
        # Run continuously in separate thread
        while True:

            # Get feedback vector if there is one
            try:
                payload = queuein.get_nowait()
                payload = payload * int(self.BLOCKSIZE/len(payload))
            except:
                payload = _filler

            # Send/receive for audio sample
            _block[:self.BLOCKSIZE] = self.spi.xfer(payload)[:self.BLOCKSIZE]

            # Fill queue
            try:
                queueout.put(_block, block=False)
            except:
                pass

                
    # Collect sample
    def transmit(self):
    
        _last = 0
        while _last <= self.bytes - self.BLOCKSIZE:
            
            _range = range(_last,
                           self.bytes - self.BLOCKSIZE+1, #don't overfill
                           self.BLOCKSIZE)
                           
            # Gather samples for instance
            for x in _range:
                self._instance[x:x+self.BLOCKSIZE] = self.audio_queue.get()
            
            # Remove filler messages
            _idxs = np.where( self._instance!=0x0055 )[0]
            self._instance[:len(_idxs)] = self._instance[_idxs]
            self._instance[len(_idxs):] = 0x0055
            _last = len(_idxs)
            
        # Convert 2 bytes to 1 audio sample
        self._instance[self._even] = np.left_shift(self._instance[self._even], 8)
        self.instance = np.bitwise_or(self._instance[self._even],
                                      self._instance[self._odd])

        # remove filler messages
        _idxs = np.where( self.instance==0x5555 )[0]
        self.instance = np.delete(self.instance, _idxs)
        
        return self.instance
    
    # Update fb payload
    def put_payload(self, payload):
        try:
            self.fb_queue.put(payload, block=False)
        except:
            print("FB queue full!")

    def __del__(self):
        self.spi.close() # close the port before exit

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    import data_processing.audio_processing as ap
    from timeit import default_timer as timer
    import traceback

    # Choose instance size to return
    sample_rate = 44100
    instance_samples = int(sample_rate*ap.INSTANCE_SIZE) # INSTANCE_SIZE = seconds
    size = int(instance_samples/2)
    
    # Instantiate communicator
    comm = audioSPI(size)

    # Prep vector to send
    vector = [1]+[0]*41
    spi_vector = np.ones(48, dtype=np.int8)
    spi_vector[:len(vector)] = np.asarray(vector, dtype=np.int8)
    spi_vector = np.packbits(spi_vector)

    # Obtain and print samples until interrupted
    try:
        while True:
            beg = timer()

            # Get instance
            comm.put_payload(spi_vector.tolist())
            instance = comm.transmit()

            # Visualize
            plt.plot(instance); plt.draw(); plt.pause(.0001)

            # Print
            print( timer()-beg, ":", min(instance), max(instance), size-len(instance))
            #import pdb;pdb.set_trace()

    except KeyboardInterrupt:
        traceback.print_exc()
        exit()
        

