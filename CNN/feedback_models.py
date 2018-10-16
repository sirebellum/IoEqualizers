import tensorflow as tf
from autoencoders import conv
BETA = 0.0005 # L2 Beta

# Get image size data based on processing
from data_processing import audio_processing as ap
HEIGHT = ap.HEIGHT
WIDTH = ap.WIDTH

NUMCLASSES = 2

def FeedbackNet(input, weights, feature_extractor):

    # Get high level features
    feedback_map = conv.encode(input, feature_extractor, weights=weights)
    
    # Deepen to low level features
    feature_map = conv.encode(feedback_map, conv.feature_encoder)
    
    # Post image of feature map
    feature_map0 = tf.slice(feature_map, [0, 0, 0, 0], [-1, -1, -1, 3])
    feature_map1 = tf.slice(feature_map, [0, 0, 0, 3], [-1, -1, -1, 3])
    feature_image = tf.concat([feature_map0, feature_map1], 2)
    tf.summary.image(
        "feature_map",
        feature_image,
        max_outputs=18
        )

    # Final feature map dimensions
    _, height, width, depth = feature_map.get_shape()
    print("CNN with final feature maps:", height, "x", width, "x", depth)
    print(height*width*depth, "total features")

    # Flattened for dense layer
    final_flat = tf.reshape(feature_map, [-1, height * width * depth])

    # Logits Layer
    logits = tf.layers.dense(
        inputs=final_flat,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
        units=NUMCLASSES)

    return logits

def model(features, labels, mode, params):

    # Input Layer
    print("Mode:", mode)
    input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, 1], name="image_input")

    # High level feature extractor (function)
    feature_extractor = params['feature_extractor']

    # Pretrained weights
    weights = params['weights']

    # Extract binary class and bins
    logits = FeedbackNet(input_layer, weights, feature_extractor)

    # If predict, return logits!
    if mode == tf.estimator.ModeKeys.PREDICT:
        return logits

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
            max_outputs=18
        )

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # L2 Regularization for logits
    loss += tf.reduce_mean(tf.losses.get_regularization_losses())

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Mask non-feedback values if classifying for feedback
    #class_mask = None
    #if NUMCLASSES == 2:
    #  class_mask = labels

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "mean_accuracy": tf.metrics.mean_per_class_accuracy(
            labels=labels, predictions=predictions["classes"],
            num_classes=NUMCLASSES),
        "recall": tf.metrics.recall(
            labels=labels, predictions=predictions["classes"]),
        "precision": tf.metrics.precision(
            labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
