# Add top level of git to path
import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np

from data_processing import audio_processing as ap
HEIGHT = ap.HEIGHT
WIDTH = ap.WIDTH
      
def parse_record(serialized_example): #parse a single binary example
    """Parses a single tf.Example into image and label tensors."""
    tfcontent = {'X': tf.FixedLenFeature([HEIGHT*WIDTH,], tf.float32),
                 'fb': tf.FixedLenFeature([1,], tf.int64),
                 'freqs': tf.FixedLenFeature([42,], tf.int64),
                 'max': tf.FixedLenFeature([1,], tf.int64),
                 }
    feature = tf.parse_single_example(serialized_example, tfcontent)

    image = tf.reshape(feature['X'], (WIDTH, HEIGHT))
    
    # fb classification labels go in labels
    labels = feature['fb'][0]
    
    # other labels and images as features
    features = {'image': image}
    for key in feature:
        if key != 'X': features[key] = feature[key]
    
    return (features, labels)
                        
# get weights from checkpoint at weights                        
def get_weights(weights):

    # Read ckpt
    checkpoint_path = weights
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Names of variables
    variables = [key for key in var_to_shape_map]
    # Only get important layers, marked with "-" at the ends
    variables = [var for var in variables if "-" in var]
    # Get actual layer names (instead of variable names)
    variables = set([var.split("-")[0] for var in variables])
    # Sort by sequence in model, first to last
    from natsort import natsorted, ns
    variables = natsorted(variables, key=lambda y: y.lower())
    
    # Collect various weights for layers
    conv_biases = [np.asarray( reader.get_tensor(key+"-/bias") ) \
                                for key in variables]
    conv_kernels = [np.asarray( reader.get_tensor(key+"-/kernel") ) \
                                for key in variables]
    
    # Aggregate variables
    variables = {}
    variables['conv_biases'] = conv_biases
    variables['conv_kernels'] = conv_kernels
        
    return variables