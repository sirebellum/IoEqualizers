import tensorflow as tf
import numpy as np
      
def parse_record(serialized_example): #parse a single binary example
  """Parses a single tf.Example into image and label tensors."""
  features = {'X': tf.FixedLenFeature([1, 12544], tf.float32)}
  feature = tf.parse_single_example(serialized_example, features)
  
  image = tf.reshape(feature['X'], (112, 112))
  
  return (image)
                        
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