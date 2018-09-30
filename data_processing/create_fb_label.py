import os
import csv

# Function meant to be used to parse seconds
def create_entry(wavfile, beg, end):
  try:
    Beginning = beg.split(":")
    Ending = end.split(":")
    
    # Invalid
    if Beginning[0] == '':
        return None
    # Only seconds
    elif len(Beginning) == 1:
        h = 0; hEnd = 0
        m = 0; mEnd = 0
        s = Beginning[0]; sEnd = Ending[0]
    # Not hours
    elif len(Beginning) == 2:
        h = 0; hEnd = 0
        m = Beginning[0]; mEnd = Ending[0]
        s = Beginning [1]; sEnd = Ending[1]
    # Everything
    else:
        h = Beginning[0]; hEnd = Ending[0]
        m = Beginning[1]; mEnd = Ending[1]
        s = Beginning [2]; sEnd = Ending[2]

    # Convert from strings to seconds
    h = int(h)*60*60
    m = int(m)*60
    s = float(s)
    hEnd = int(hEnd)*60*60
    mEnd = int(mEnd)*60
    sEnd = float(sEnd)
    
    start = h+m+s
    duration = hEnd+mEnd+sEnd - start
    
    print("Feedback for {} seconds".format(duration))
    
    return [wavfile, start, duration]
    
  except ValueError:
    return None
if __name__ == "__main__":
    
    
    labelfile = input("What would you like to name this session?: ")
    labelfile = labelfile+".csv"
    wavfile = input("What is the name of the wav file?: ")
    entries = list()
    
    # Begin entries
    while True:
        try:
            while True:
                beg = input("Beginning time of feedback (hh:mm:ss.s)?: ")
                end = input("End time of feedback (hh:mm:ss.s)?: ")
                entries.append(create_entry(wavfile, beg, end))
                if entries[-1] is None:
                    entries.pop()
                    print("Invalid entry!")
            
        # Check for next wav file
        except KeyboardInterrupt:
            wavfile = input("\nWhat is the name of the wav file (n for none)?: ")
            if wavfile != "n": continue
            
            # Write csv file if none left
            with open(labelfile, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.   QUOTE_MINIMAL)
                
                for entry in entries:
                    csv_writer.writerow(entry)
                    
            exit("Written to {}!".format(labelfile))
