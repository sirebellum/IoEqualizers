import tensorflow as tf
import random
import math
import numpy as np
import audio_processing as ap
import os
import gc

def label_number(label_string): #turn string into integer
  if "dark" in label_string:
    return 0
  elif "bright" in label_string: 
    return 1
  else:
    return 2

def _list_feature(value):
  return tf.train.Feature(int64_list =tf.train.Int64List(value=value.reshape(-1)))

def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
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
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)            
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print( "Serializing {:d} examples into {}".format(X.shape[0], result_tf_file) )
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        
        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)
            
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
    num_instances = 12800
    
    # Nsynth dataset
    dataset = ap.nsynth(dataset_dir)

    batch = dataset.returnInstance(num_instances)
    while batch is not None:
        #encoded_jpg = cv2.imencode(".jpg", image)[1].tostring()
        images = [ ap.plotSpectrumBW(image).flatten() for image in batch['fft'] ]
        images = np.asarray(images, dtype=np.float32)
        images *= 1/255.0
        
        # Labels
        instrument_source = [ label for label in batch['instrument_source'] ]
        instrument_source = np.asarray(instrument_source, dtype=np.int64)
        instrument_family = [ label for label in batch['instrument_family'] ]
        instrument_family = np.asarray(instrument_family, dtype=np.int64)
        # Aggregate labels
        labels = list( zip(instrument_source, instrument_family) )
        labels = np.asarray(labels)
        
        import ipdb; ipdb.set_trace()
        
        np_to_tfrecords(images, labels, path_to_write+str(shard))
        
        # Cleanup to conserve RAM
        del images
        del batch
        gc.collect()
        
        batch = dataset.returnInstance(num_instances)
        shard += 1
        
def main(_):

    ###Create valid tfrecord###
    path_to_write = os.path.join(os.getcwd()) + '/nsynth/valid_labeled'
    dataset_dir = "nsynth/nsynth-valid/"
    generate(path_to_write, dataset_dir)
        
        
    ###Create Train tfrecords###
    path_to_write = os.path.join(os.getcwd()) + '/nsynth/train_labeled'
    dataset_dir = "nsynth/nsynth-train/"
    generate(path_to_write, dataset_dir)


if __name__ == '__main__':
    tf.app.run()