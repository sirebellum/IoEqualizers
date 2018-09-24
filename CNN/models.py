import tensorflow as tf
from autoencoders import conv
extract = conv.encode
HEIGHT = 112
WIDTH = 112
BETA = 0.001 # L2 Beta

def cnn(input_layer, weights):
  """Model function for CNN."""
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[5, 5],
      strides=(1, 1),
      padding="valid",
      activation=tf.nn.relu)
  
  # Pool Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=(1,1))
  
  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      strides=(1, 1),
      padding="valid",
      activation=tf.nn.relu)
      
  # Pool Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=(1,1))
  
  # Pool Layer #3
  pool3 = tf.layers.max_pooling2d(inputs=pool2, pool_size=[2, 2], strides=(1,1))
  
  # Final layer for conversion
  final = pool3
  
  return final

def classifier(features, labels, mode, params):

  # Input Layer
  print("Mode:", mode)
  input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, 1], name="image_input")

  # For classification purposes
  NUMCLASSES = params['num_labels']

  # Feature extractor (function)
  feature_extractor = params['feature_extractor']
  
  # Pretrained weights
  weights = params['weights']
  
  # Extract final layer for classification
  feature_map = extract(input_layer, weights, feature_extractor)

  # Final feature map dimensions
  _, height, width, depth = feature_map.get_shape()
  print("CNN with final feature maps:", height, "x", width, "x", depth)
  print(height*width*depth, "total features")
  
  # Dense layer
  final_flat = tf.reshape(feature_map, [-1, height * width * depth])
  dropout = tf.layers.dropout(
      inputs=final_flat, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(
    inputs=dropout,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
    units=NUMCLASSES)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  # Put images in tensorboard
  if mode == tf.estimator.ModeKeys.TRAIN:
      tf.summary.image(
        "Image",
        input_layer,
        max_outputs=9
      )
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # L2 Regularization for layer weights
  loss = tf.reduce_mean(loss + tf.losses.get_regularization_loss())

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "mean_accuracy": tf.metrics.mean_per_class_accuracy(
          labels=labels, predictions=predictions["classes"],
          num_classes=NUMCLASSES)
  }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
