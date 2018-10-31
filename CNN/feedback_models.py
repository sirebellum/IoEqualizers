import sys
sys.path.append('../') # Top directory

import tensorflow as tf
from autoencoders import conv
BETA = 0.0025 # L2 Beta

# Get image size data based on processing
from data_processing import audio_processing as ap
HEIGHT = ap.HEIGHT
WIDTH = ap.WIDTH

NUMCLASSES = 2

def FeedbackNet(input, weights, feature_extractor):

    # Get high level features
    frequency_map = conv.encode(input, feature_extractor, weights=weights)
    
    ### Feedback classification
    # Deepen to low level features
    feature_map = conv.encode(frequency_map, conv.feature_encoder)
    
    # Post image of feature map
    feature_map0 = tf.slice(feature_map, [0, 0, 0, 0], [-1, -1, -1, 3])
    feature_map1 = tf.slice(feature_map, [0, 0, 0, 3], [-1, -1, -1, 3])
    feature_image = tf.concat([feature_map0, feature_map1], 2)
    tf.summary.image(
        "feature_map",
        feature_image,
        max_outputs=18
        )
    
    # Deep feature map dimensions
    _, height, width, depth = feature_map.get_shape()
    print("CNN with low level feature map:", height, "x", width, "x", depth)
    print(height*width*depth, "total features")
    '''
    # Flattened for dense layer
    final_flat = tf.reshape(feature_map, [-1, height * width * depth])

    # Logits Layer
    logits = tf.layers.dense(
        inputs=final_flat,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
        units=NUMCLASSES)
    '''
    
    ### Feedback localization
    ### TODO: Batch aware feedback vector creation
    ###  Loss function that takes false negative classifications into account.
    ###     Maybe don't add loss on negative classifications?
    # Whether or not feedback is present
    #fb_bool = tf.cast(tf.argmax(input=logits, axis=1), tf.bool)
    
    # Get freq vector
    freq_vector = conv.encode(feature_map, conv.fb_vectorize)
    
    _, freqs, time, filters = freq_vector.get_shape()
    print("Feedback vector map:", freqs, "x", time, "x", filters)
    print(freqs*time*filters, "total features")
    
    # Flatten in prep for dense layer
    freq_vector = tf.reshape(freq_vector, [-1, freqs * time * filters])
    
    # Dense layer for predicting actual feedback bins
    logits = tf.layers.dense(
        inputs=freq_vector,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
        units=freqs)
    
    return logits

def model(features, labels, mode, params):
    # features breakout
    image = features['image']
    freq_labels = features['freqs'] # frequency vector
    max = features['max']     # max freq in vector
    
    # PNG processing
    if image.dtype == tf.string:
        image = tf.map_fn(tf.decode_base64, image)
        image = tf.map_fn(tf.image.decode_png, image, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)
        image /= 255.0

    # Input Layer
    print("Mode:", mode)
    input_layer = tf.reshape(image, [-1, HEIGHT, WIDTH, 1], name="image_input")

    # High level feature extractor (function)
    feature_extractor = params['feature_extractor']

    # Pretrained weights
    weights = params['weights']

    # Extract binary class and bins
    logits = FeedbackNet(input_layer, weights, feature_extractor)

    # If predict, return logits!
    if mode == tf.estimator.ModeKeys.PREDICT:
        return logits

    # Put images in tensorboard
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.image(
            "Image",
            input_layer,
            max_outputs=18
        )
        
    # Calculate classification Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=freq_labels, logits=logits)

    # L2 Regularization for logits
    loss += tf.reduce_mean(tf.losses.get_regularization_losses())

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Get freq vector
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        #"classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`
        "freq_probs": tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        "freq_vector": tf.to_int64(tf.nn.sigmoid(logits, name="sigmoid_tensor") > 0.9)
    }
    
    # Adapts metric functions with non scalar outputs
    def metric_fn(fn, labels, predictions, threshold):
        value, update_op = fn(labels=labels, predictions=predictions, thresholds=[threshold])
        return tf.squeeze(value), update_op
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
                        labels=freq_labels,
                        predictions=predictions['freq_vector']
                        ),
        "0.5@recall": metric_fn(
                        tf.metrics.recall_at_thresholds,
                        labels=freq_labels,
                        predictions=predictions['freq_probs'],
                        threshold=0.5
                        ),
        "0.5@precision": metric_fn(
                        tf.metrics.precision_at_thresholds,
                        labels=freq_labels,
                        predictions=predictions['freq_probs'],
                        threshold=0.5
                        ),
        "0.7@recall": metric_fn(
                        tf.metrics.recall_at_thresholds,
                        labels=freq_labels,
                        predictions=predictions['freq_probs'],
                        threshold=0.7
                        ),
        "0.7@precision": metric_fn(
                        tf.metrics.precision_at_thresholds,
                        labels=freq_labels,
                        predictions=predictions['freq_probs'],
                        threshold=0.7
                        ),
        "0.9@recall": metric_fn(
                        tf.metrics.recall_at_thresholds,
                        labels=freq_labels,
                        predictions=predictions['freq_probs'],
                        threshold=0.9
                        ),
        "0.9@precision": metric_fn(
                        tf.metrics.precision_at_thresholds,
                        labels=freq_labels,
                        predictions=predictions['freq_probs'],
                        threshold=0.9
                        )
    }
    ''' Classification metrics
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
    '''
    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
