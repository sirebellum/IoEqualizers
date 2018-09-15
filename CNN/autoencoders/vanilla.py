import tensorflow as tf

def encode(features, labels, mode, params):

  # Input Layer
  print("Mode:", mode)
  input_layer = tf.reshape(features, [-1, 28, 28, 1], name="image_input")
  
  # Hidden Layer
  flattened = tf.reshape(input_layer, [-1, 784])
  hidden_layer = tf.layers.dense(inputs=flattened, units=64, activation=tf.nn.relu)
  
  # Decoding Layer
  output_layer = tf.layers.dense(inputs=hidden_layer, units=784, activation=tf.nn.relu)
  
  # Reshape to image
  reconstructed = tf.reshape(output_layer, [-1, 28, 28, 1], name="image_output")
  
  # Calculate Loss
  loss = tf.losses.mean_squared_error(labels=flattened,
                                      predictions=output_layer)
                                   
  # Put images in tensorboard
  if mode == tf.estimator.ModeKeys.TRAIN:
      tf.summary.image(
        "Image",
        reconstructed,
        max_outputs=18
      )                
                
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(epsilon=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)