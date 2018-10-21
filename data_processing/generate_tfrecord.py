import tensorflow as tf
import random
import math
import numpy as np
import datasets
import os
import glob

def _list_feature(value):
  return tf.train.Feature(int64_list =tf.train.Int64List(value=value.reshape(-1)))

def np_to_tfrecords(X, file_path_prefix, verbose=True, **kwargs):
    """
    Converts a Numpy arrays into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to other kwargs.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    kwargs : numpy.ndarray(s) of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instead got {}".format(ndarray.dtype))
            
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    for key in kwargs:
        assert isinstance(kwargs[key], np.ndarray) or kwargs is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    dtype_feature_y = {}
    for key in kwargs:
        assert X.shape[0] == kwargs[key].shape[0]
        assert len(kwargs[key].shape) == 2
        dtype_feature_y[key] = _dtype_feature(kwargs[key])            
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print( "Serializing {:d} examples into {}".format(X.shape[0], result_tf_file) )
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        y = {}
        for key in kwargs:
            y[key] = kwargs[key][idx]
        
        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        for key in kwargs:
            d_feature[key] = dtype_feature_y[key](y[key])
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print( "Writing {} done!".format(result_tf_file) )


def generate(path_to_write, dataset_dir):
    
    # Sharts to split into multiple files
    shard = 0
    
    # Instances to write at a time
    num_instances = 2560
    
    # Nsynth dataset
    #dataset = ap.nsynth(dataset_dir)

    # Feedback dataset
    feedback_files = glob.glob(dataset_dir+"*.csv")
    dataset = datasets.feedback(feedback_files, self_sample=True)
    
    batch = dataset.returnInstance(num_instances)
    while batch is not None:
        #images = [ ap.plotSpectrumBW(image).flatten() for image in batch['fft'] ]
        images = [ image.flatten() for image in batch['fft'] ]
        images = np.asarray(images, dtype=np.float32)
        images *= 1/255.0
        
        # Labels
        fb = [ label for label in batch['fb'] ]
        fb = np.asarray(fb, dtype=np.int64)
        freqs = [ label for label in batch['freqs_vector'] ]
        freqs = np.asarray(freqs, dtype=np.int64)
        max = [ label for label in batch['max'] ]
        max = np.asarray(max, dtype=np.int64)
        # Match what tfrecord writer is expecting
        fb = np.expand_dims(fb, 1)
        max = np.expand_dims(max, 1)
        
        # Write tf records
        np_to_tfrecords(
            images,
            path_to_write+str(shard),
            fb=fb,
            freqs=freqs,
            max=max,
        )
        
        batch = dataset.returnInstance(num_instances)
        shard += 1
        
    del dataset
    del batch
    del images
        
def main(_):

    ###Create valid tfrecord###
    #path_to_write = os.path.join(os.getcwd()) + '/nsynth/valid'
    #dataset_dir = "nsynth-valid/"
    #generate(path_to_write, dataset_dir)
        
        
    ###Create Train tfrecords###
    #path_to_write = os.path.join(os.getcwd()) + '/nsynth/train'
    #dataset_dir = "nsynth-train/"
    #generate(path_to_write, dataset_dir)
    
    ###Create Feedback tfrecords###
    path_to_write = os.path.join(os.getcwd()) + '/tfrecords/feedback'
    dataset_dir = "feedback/"
    generate(path_to_write, dataset_dir)

if __name__ == '__main__':
    tf.app.run()