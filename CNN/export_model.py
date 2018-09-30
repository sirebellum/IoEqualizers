import sys
sys.path.append('../')

import tensorflow as tf
import class_models
# Autoencoders
from autoencoders import conv, vanilla

# which model to use
feature_extractor = conv.conv_instrument

slim = tf.contrib.slim
import os
import json

tf.app.flags.DEFINE_integer('model_version', 1, 'Models version number.')
tf.app.flags.DEFINE_string('work_dir', './models', 'Working directory.')
tf.app.flags.DEFINE_string('model_id', "feedback", 'Model id name to be loaded.')
tf.app.flags.DEFINE_string('export_model_dir', "./deploy", 'Directory where the model exported files should be placed.')

FLAGS = tf.app.flags.FLAGS

model_name = str(FLAGS.model_id)
log_folder = FLAGS.work_dir

def main(_):

    with tf.Session(graph=tf.Graph()) as sess:
        # define placeholders for receiving the input image height and width
        image_height_tensor = tf.placeholder(tf.int32)
        image_width_tensor = tf.placeholder(tf.int32)

        # placeholder for receiving the serialized input image
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {'x': tf.FixedLenFeature(shape=[], dtype=tf.float32), }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)

        # reshape the input image to its original dimension
        tf_example['x'] = tf.reshape(tf_example['x'], (1, image_height_tensor, image_width_tensor, 1))
        input_tensor = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name

        # Define params for model
        params = {}
        params['num_labels'] = 2
        params['feature_extractor'] = feature_extractor
        params['weights'] = None
        
        # perform inference on the input image
        logits_tf = class_models.classifier(input_tensor, None, tf.estimator.ModeKeys.PREDICT, params)

        # extract the classifications
        predictions_tf = tf.argmax(logits_tf, axis=1)

        # specify the directory where the pre-trained model weights are stored
        pre_trained_model_dir = os.path.join(log_folder, model_name)

        saver = tf.train.Saver()

        # Restore variables from disk.
        recent_model = tf.train.latest_checkpoint(pre_trained_model_dir)
        saver.restore(sess, recent_model)
        print("Model", model_name, "restored.")

        # Create SavedModelBuilder class
        # defines where the model will be exported
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(model_name),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
        tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
        tensor_info_height = tf.saved_model.utils.build_tensor_info(image_height_tensor)
        tensor_info_width = tf.saved_model.utils.build_tensor_info(image_width_tensor)

        # output tensor info
        tensor_info_output = tf.saved_model.utils.build_tensor_info(predictions_tf)

        # Defines the DeepLab signatures, uses the TF Predict API
        # It receives an image and its dimensions and output the segmentation mask
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input, 'height': tensor_info_height, 'width': tensor_info_width},
                outputs={'segmentation_map': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            })

        # export the model
        builder.save(as_text=True)
        print('Done exporting!')

if __name__ == '__main__':
    tf.app.run()