''' records audio from specified input and creates label file'''
import wave
import numpy as np
import pyaudio
import datetime
from multiprocessing import Process, Queue
import os
import csv

# Function meant to be used by second thread to write
#label entry without interrupting the main thread.
def create_entry(begin, now):
    
    entry = now - begin
    return [entry.total_seconds()]

# Record main thread
def record():
    # Wav recording properties
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    
    # Filename and directory setup
    now = datetime.datetime.now()
    DATE = now.strftime("%Y-%m-%d")
    WAVE_DIR = input("Directory for saving (recordings)?: ")
    if WAVE_DIR == "":
        WAVE_DIR="recordings" # Default
    WAVE_DIR = WAVE_DIR+"_"+DATE+"/"
    if not os.path.exists(WAVE_DIR):
        os.makedirs(WAVE_DIR)
    else: exit("Directory exists!")
    
    # Open pyaudio interface
    p = pyaudio.PyAudio()
    
    # Interactive input select
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    device_index = int( input("What device index would you like to use?: ") )

    # Set up recording stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index)
                    
    # Begin recording
    print("* recording")
    FILE_INDEX = 0 # Name each subsequent file
    try:
      while True:
        # Record for RECORD_SECONDS
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        # Write Wav file
        print("Writing file...")
        FILENAME = WAVE_DIR+str(FILE_INDEX)+".wav"
        wf = wave.open(FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print("Wrote {}".format(FILENAME))
        
        FILE_INDEX += 1

    except KeyboardInterrupt:
        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    
    
    labelfile = input("What would you like to name this session?: ")
    labelfile = labelfile+".csv"
    entries = list()
    input("Press enter when ready to start labeling...\n")
    beginning = datetime.datetime.now()
    print("Labeling!...")
    try:
        while True:
            user_input = input()
            if "r" in user_input:
                print("Removing {} entry!".format(entries.pop()))
                continue
            now = datetime.datetime.now()
            entries.append(create_entry(beginning, now))
            
    
    # Write out entries
    except KeyboardInterrupt:
        with open(labelfile, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.   QUOTE_MINIMAL)
            
            for entry in entries:
                csv_writer.writerow(entry)