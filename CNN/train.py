# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
import data_processing.audio_processing as ap
from functions import parse_record, get_weights
import models
import glob

# Autoencoders
from autoencoders import conv, vanilla

# which model to use
feature_extractor = conv.encode3x3

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
parser.add_argument("--steps", default=5000, help="Train to number of steps")
parser.add_argument("--weights", default=None, help="Model checkpoint to get pretrained weights from")
args = parser.parse_args()
num_steps = int(args.steps)

# Directory setup
abs_path = os.path.abspath(__file__) # Absolute path of this file
directory = os.path.dirname(abs_path)
model_dir = directory+"/models/"+args.output_name

# Get pretrained weights for feature extractor
weights = None
if args.weights is not None:
    weights = os.path.join(os.path.dirname(__file__), args.weights)
    weights = get_weights(weights) # numpy weights

# Define the input function for training
def tfrecord_input():

  # Keep list of filenames, so you can input directory of tfrecords easily
  train_filenames = glob.glob("data_processing/nsynth_ssd/train*tfrecords")
  valid_filenames = glob.glob("data_processing/nsynth_ssd/valid*tfrecords")
  batch_size=256

  # Import data
  dataset = tf.data.TFRecordDataset(
        train_filenames,
        num_parallel_reads=6,
        buffer_size=1000*1000*128) # 128mb of io cache
        
  ### IMPLEMENT SHUFFLE ###

  # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(parse_record, num_parallel_calls=6)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(batch_size*4) # Prefetch 4 batches at a time
  #print("Dataset:", dataset.output_shapes, ":::", dataset.output_types)
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  #print("Iterator:", features)

  return (features, labels)
  

def main(unused_argv):
  
    # input
    train_input_fn = tfrecord_input
  
    # Define params for model
    params = {}
    params['num_labels'] = 11
    params['feature_extractor'] = feature_extractor
    #params['noise'] = True
    params['weights'] = weights

    # Reduce GPU memory allocation
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    
    # Estimator config to change frequency of ckpt files
    estimator_config = tf.estimator.RunConfig(
        session_config = sess_config,
        save_checkpoints_secs = 60*5,  # Save checkpoints every 5 minutes
        keep_checkpoint_max = 2)       # Retain the 2 most recent checkpoints.

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=models.classifier,
        model_dir=model_dir,
        config=estimator_config,
        params=params)
    
    classifier.train(
        input_fn=train_input_fn,
        steps=num_steps)


if __name__ == "__main__":
  tf.app.run()
