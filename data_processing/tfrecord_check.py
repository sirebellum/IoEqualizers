import tensorflow as tf
import numpy as np
import matplotlib
import glob
from PIL import Image

filenames = glob.glob("tfrecords/*tfrecords")
# 1-2. Check if the data is stored correctly
# open the saved file and check the first entries
for filename in filenames:
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        x_1 = np.array(example.features.feature['X'].float_list.value)
        fb = np.array(example.features.feature['fb'].int64_list.value)
        max = np.array(example.features.feature['max'].int64_list.value)
        freqs = np.array(example.features.feature['freqs'].int64_list.value)
        
        x_1 = np.reshape(x_1, (168, 56))
        x_1 *= 255.0
        x_1 = x_1.astype(np.uint8)
        
        if len(freqs) != 36:
            print(fb, max, len(freqs))
            img = Image.fromarray(x_1, 'L')
            img.show()

            input()