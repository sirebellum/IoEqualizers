import os
import csv

# Function meant to be used to parse seconds
def create_entry(wavfile, beg, end):
    
    h, m, s = beg.split(":")
    hEnd, mEnd, sEnd = end.split(":")
    
    # Convert from strings to seconds
    h = int(h)*60*60
    m = int(m)*60
    s = int(s)
    hEnd = int(hEnd)*60*60
    mEnd = int(mEnd)*60
    sEnd = int(sEnd)
    
    start = h+m+s
    duration = hEnd+mEnd+sEnd - start
    
    print("Feedback for {} seconds".format(duration))
    
    return [wavfile, start, duration]

if __name__ == "__main__":
    
    
    labelfile = input("What would you like to name this session?: ")
    labelfile = labelfile+".csv"
    wavfile = input("What is the name of the wav file?: ")
    entries = list()
    
    # Begin entries
    while True:
        try:
            while True:
                beg = input("Beginning time of feedback (hh:mm:ss)?: ")
                end = input("End time of feedback (hh:mm:ss)?: ")
                entries.append(create_entry(wavfile, beg, end))
            
        # Check for next wav file
        except KeyboardInterrupt:
            wavfile = input("\nWhat is the name of the wav file (n for none)?: ")
            if wavfile != "n": continue
            
            # Write wav file if none left
            with open(labelfile, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.   QUOTE_MINIMAL)
                
                for entry in entries:
                    csv_writer.writerow(entry)
                    
            exit("Written to {}!".format(labelfile))