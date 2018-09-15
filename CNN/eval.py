# Imports
import numpy as np
import os
import argparse
import inotify.adapters
import tensorflow as tf
import data_processing.audio_processing as ap
import glob

# Autoencoders
from autoencoders import conv, vanilla

# Which model to use
cnn_model = conv.encode

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Relative path to model")
parser.add_argument("--eval", default=0, help="Evaluate only the most recent checkpoint if set")
args = parser.parse_args()

#Allow either model name or directory to be used
CWD_PATH = os.getcwd()
if "models" and "/" not in args.output_name:
  model_path = CWD_PATH+"/models/"+args.output_name
else:
  model_path = CWD_PATH+"/"+args.output_name
print("Set to evaluate model at", model_path)

#Inotify setup
file_watch = inotify.adapters.Inotify()
file_watch.add_watch(model_path)

def tfrecord_input():

    # Keep list of filenames, so you can input directory of tfrecords easily
    train_filenames = glob.glob("data_processing/nsynth/train*tfrecords")
    valid_filenames = glob.glob("data_processing/nsynth/valid*tfrecords")
    batch_size=256

    # Import data
    dataset = tf.data.TFRecordDataset(valid_filenames)

    # Map the parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(parse_record)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1) # Only go over validation set once
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
    eval_input_fn = tfrecord_input

    # Define params for model
    params = {}
    params['num_labels'] = 0

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model,
        model_dir=model_path,
        params=params)

    # Evaluate immediately
    print("Evaluating...")
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    if args.eval: # exit if flag set
        exit() 

    # Evaluate for every new file
    for event in file_watch.event_gen(yield_nones=False):
        # Evaluate the model and print results
        (_, type_names, path, filename) = event
        new_ckpt = type_names[0] is 'IN_MOVED_TO' and 'checkpoint' in filename and 'tmp' not in filename
        if new_ckpt:
          print("Evaluating...")
          eval_results = classifier.evaluate(input_fn=eval_input_fn)
          print(eval_results)
  
if __name__ == "__main__":
  tf.app.run()
