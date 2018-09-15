# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
import data_processing.audio_processing as ap

# Autoencoders
from autoencoders import conv, vanilla

# which model to use
cnn_model = conv.encode

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
parser.add_argument("--steps", default=5000, help="Train to number of steps")
args = parser.parse_args()
num_steps = int(args.steps)

CWD_PATH = os.getcwd()

def main(unused_argv):

    # Set up data
    train_data = ap.nsynth("nsynth/nsynth-train/")
  
    # Define params for model
    params = {}
    params['num_labels'] = 1

    # Estimator config to change frequency of ckpt files
    estimator_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 10,  # Save checkpoints every 10 seconds
    keep_checkpoint_max = 2)       # Retain the 2 most recent checkpoints.

    # Create the Estimator
    classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=CWD_PATH+"/models/"+args.output_name,
    config=estimator_config,
    params=params)
  
    # Train
    train_images = [0]
    while len(train_images) > 0:
  
        batch = train_data.returnInstance(12800)
        train_images = [ ap.plotSpectrumBW(image) for image in batch['fft'] ]
        train_images = np.asarray(train_images, dtype=np.float16)
        train_images *= 1/255.0
  
        train_input_fn = \
               tf.estimator.inputs.numpy_input_fn(train_images,
                                                  batch_size=128,
                                                  num_epochs=1,
                                                  shuffle=True,
                                                  queue_capacity=1024,
                                                  num_threads=2)

        # Set up logging for predictions
        #tensors_to_log = {"predictions": "image_output"}
        #logging_hook = tf.train.LoggingTensorHook(
        #    tensors=tensors_to_log, every_n_iter=50)
          
        # Train the model
        classifier.train(
            input_fn=train_input_fn,
            steps=None)
            
        del batch
        del train_images

if __name__ == "__main__":
  tf.app.run()
