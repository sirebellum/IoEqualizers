# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
import data_processing.audio_processing as ap
import glob

# Autoencoders
from autoencoders import conv, vanilla, convnoise

# which model to use
cnn_model = convnoise.encode

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
parser.add_argument("--steps", default=5000, help="Train to number of steps")
args = parser.parse_args()
num_steps = int(args.steps)

CWD_PATH = os.getcwd()

# Define the input function for training
def tfrecord_input():

  # Keep list of filenames, so you can input directory of tfrecords easily
  train_filenames = glob.glob("data_processing/nsynth/train*tfrecords")
  valid_filenames = glob.glob("data_processing/nsynth/valid*tfrecords")
  batch_size=256

  # Import data
  dataset = tf.data.TFRecordDataset(
        train_filenames,
        num_parallel_reads=4,
        buffer_size=1000*1000*128) # 128mb of io cache

  # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(parse_record, num_parallel_calls=4)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(batch_size*2) # Prefetch 2 batches at a time
  #print("Dataset:", dataset.output_shapes, ":::", dataset.output_types)
  iterator = dataset.make_one_shot_iterator()

  features = iterator.get_next()
  #print("Iterator:", features)

  return (features)
  
def parse_record(serialized_example): #parse a single binary example
  """Parses a single tf.Example into image and label tensors."""
  features = {'X': tf.FixedLenFeature([1, 12544], tf.float32)}
  feature = tf.parse_single_example(serialized_example, features)
  
  image = tf.reshape(feature['X'], (112, 112))
  
  return (image)
  

def main(unused_argv):
  
    # input
    train_input_fn = tfrecord_input
  
    # Define params for model
    params = {}
    params['num_labels'] = 0

    # Estimator config to change frequency of ckpt files
    estimator_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 60*5,  # Save checkpoints every 5 minutes
    keep_checkpoint_max = 2)       # Retain the 2 most recent checkpoints.

    # Create the Estimator
    classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=CWD_PATH+"/models/"+args.output_name,
    config=estimator_config,
    params=params)
    
    classifier.train(
        input_fn=train_input_fn,
        steps=num_steps)


if __name__ == "__main__":
  tf.app.run()
