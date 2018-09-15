import tensorflow as tf
import numpy as np
import matplotlib
import glob
from PIL import Image

filenames = glob.glob("nsynth/*tfrecords")
# 1-2. Check if the data is stored correctly
# open the saved file and check the first entries
for serialized_example in tf.python_io.tf_record_iterator(filenames[0]):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    x_1 = np.array(example.features.feature['X'].float_list.value)
    
    x_1 = np.reshape(x_1, (112, 112))
    x_1 *= 255.0
    x_1 = x_1.astype(np.uint8)
    
    img = Image.fromarray(x_1, 'L')
    img.show()

    input()