# Add top level of git to path
import sys
sys.path.append("../")

# Imports
import numpy as np
import os
import argparse
import inotify.adapters
import tensorflow as tf
from functions import parse_record, get_weights
import feedback_models
import glob

# Autoencoders
from autoencoders import conv, vanilla

# Which model to use
feature_extractor = conv.frequency_encoder

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Relative path to model")
parser.add_argument("--eval", default=0, help="Evaluate only the most recent checkpoint if set")
parser.add_argument("--weights", default=None, help="Model checkpoint to get pretrained weights from")
args = parser.parse_args()

# Directory setup
abs_path = os.path.abspath(__file__) # Absolute path of this file
directory = os.path.dirname(abs_path)
model_dir = os.path.join(directory, args.output_name)

# Get pretrained weights for feature extractor
weights = None
if args.weights is not None:
    weights = os.path.join(os.path.dirname(__file__), args.weights)
    weights = get_weights(weights) # numpy weights
    
#Inotify setup
file_watch = inotify.adapters.Inotify()
file_watch.add_watch(model_dir)

def tfrecord_input():

    # Keep list of filenames, so you can input directory of tfrecords easily
    train_filenames = glob.glob("../data_processing/tfrecords/*tfrecords")
    valid_filenames = glob.glob("../data_processing/tfrecords/val*tfrecords")
    batch_size=256

    # Import data
    dataset = tf.data.TFRecordDataset(valid_filenames)

    # Map the parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(parse_record)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(int(12678/batch_size)) # Only go over validation set once
    #print("Dataset:", dataset.output_shapes, ":::", dataset.output_types)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    #print("Iterator:", features)

    return (features, labels)

def main(unused_argv):

    # input
    eval_input_fn = tfrecord_input

    # Define params for model
    params = {}
    params['noise'] = True
    params['feature_extractor'] = feature_extractor
    params['weights'] = weights

    # Reduce GPU memory allocation
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    estimator_config = tf.estimator.RunConfig(session_config=sess_config)
    
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=feedback_models.model,
        model_dir=model_dir,
        config=estimator_config,
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
