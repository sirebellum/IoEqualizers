# Add top level of git to path
import sys
sys.path.append("../../")

import tensorflow as tf
import math
NOIS_MEAN = 0.0
NOISE_STD = 0.2
BETA = 0.000075

# Get image size data based on processing
from data_processing import audio_processing as ap
HEIGHT = ap.HEIGHT
WIDTH = ap.WIDTH

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=NOIS_MEAN, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def encode(features, encoder, weights=None):

    # Initialization weights
    if weights is not None:
        biases = [ tf.constant_initializer(value) for value in weights['conv_biases'] ]
        kernels = [ tf.constant_initializer(value) for value in weights['conv_kernels'] ]
        # Reverse to utilze list.pop method
        biases.reverse()
        kernels.reverse()
    else: # If no weights, set to doc defaults
        biases = [tf.zeros_initializer()] * 999
        kernels = [None] * 999
    
    # Encoder
    try:
        feature_map = encoder(features, kernels, biases)
    
    # error for popping too many weights off of list
    except IndexError:
        exit("Not enough weights for model! Make sure architectures are compatible")
    # error for mismatching kernel size
    except ValueError:
        exit("Kernel weight dimensions do not match! Make sure architectures are compatible")
    
    # Prompt if not all the weights were used
    if len(kernels) > 0 and kernels.pop() != None:
        response = input("Not all weights used! {}/{} Are you sure you want to continue?" \
                            .format( len(weights['conv_kernels']), len(kernels)+1 ) )
    
    
    return feature_map

# Encoder with kernels specifically for frequency encoding
def frequency_encoder(features, kernels, biases):

    # Check for valid weights
    restore = False
    if kernels[0] is not None:
        restore = True

    # Capture largest frequency dependent features
    conv_freq1 = tf.layers.Conv2D(
        6, (HEIGHT/2, 2), strides=(HEIGHT/4, 2),
        activation='relu', padding='same', name='conv1-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA/10),
        bias_initializer=biases.pop())(features)
    # Capture large frequency dependent features
    conv_freq2 = tf.layers.Conv2D(
        6, (HEIGHT/4, 2), strides=(HEIGHT/8, 2),
        activation='relu', padding='same', name='conv2-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA/10),
        bias_initializer=biases.pop())(features)
    # Capture small frequency dependent features
    conv_freq3 = tf.layers.Conv2D(
        6, (HEIGHT/8, 2), strides=(HEIGHT/16, 2),
        activation='relu', padding='same', name='conv3-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA/10),
        bias_initializer=biases.pop())(features)
    # Capture smallest frequency dependent features
    conv_freq4 = tf.layers.Conv2D(
        6, (HEIGHT/21, 2), strides=(HEIGHT/42, 2),
        activation='relu', padding='same', name='conv4-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA/10),
        bias_initializer=biases.pop())(features)
    
    # Pool out time scales
    pool_freq1 = tf.layers.MaxPooling2D((2, WIDTH/8), (1, WIDTH/16), padding='same', name='pool5-')(conv_freq1)
    pool_freq2 = tf.layers.MaxPooling2D((2, WIDTH/8), (1, WIDTH/16), padding='same', name='pool6-')(conv_freq2)
    pool_freq3 = tf.layers.MaxPooling2D((2, WIDTH/8), (1, WIDTH/16), padding='same', name='pool7-')(conv_freq3)
    pool_freq4 = tf.layers.MaxPooling2D((2, WIDTH/8), (1, WIDTH/16), padding='same', name='pool8-')(conv_freq4)
    '''
    # Pad smaller feature maps
    pool_freq1 = tf.pad(pool_freq1, tf.constant([[0, 0], [17, 17], [0, 0], [0, 0]]))
    pool_freq2 = tf.pad(pool_freq2, tf.constant([[0, 0], [15, 15], [0, 0], [0, 0]]))
    pool_freq3 = tf.pad(pool_freq3, tf.constant([[0, 0], [11, 11], [0, 0], [0, 0]]))
    
    # Concat into same feature map
    freq_map = tf.concat([pool_freq1, pool_freq2, pool_freq3, pool_freq4], 3)
    '''
    
    # Upscale smaller frequency maps
    _, height, width, depth = pool_freq4.get_shape()
    pool_freq1 = tf.image.resize_nearest_neighbor(pool_freq1, [height, width])
    pool_freq2 = tf.image.resize_nearest_neighbor(pool_freq2, [height, width])
    pool_freq3 = tf.image.resize_nearest_neighbor(pool_freq3, [height, width])
    
    # Concat into same feature map
    freq_map = tf.concat([pool_freq1, pool_freq2, pool_freq3, pool_freq4], 3)
    
    # Post image of feedback map BROKEN
    feedback_map = tf.concat([pool_freq1, pool_freq2, pool_freq3, pool_freq4], 2)
    feedback_image0 = tf.slice(feedback_map, [0, 0, 0, 0], [-1, -1, -1, 3])
    feedback_image1 = tf.slice(feedback_map, [0, 0, 0, 3], [-1, -1, -1, 3])
    feedback_image = tf.concat([feedback_image0, feedback_image1], 2)
    tf.summary.image(
        "feedback_map",
        feedback_image,
        max_outputs=18
        )
    
    # If valid weights were loaded
    if restore:
        # Don't update layers
        tf.stop_gradient(conv_freq1)
        tf.stop_gradient(conv_freq2)
        tf.stop_gradient(conv_freq3)
        tf.stop_gradient(conv_freq4)
    
    return freq_map

# Create deeper encoding of high level features
def feature_encoder(features, kernels, biases):

    # Check for valid weights
    restore = False
    if kernels[0] is not None:
        restore = True
    
    # Pad so we don't lose frequencies
    features = tf.pad(features, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]))
    conv1 = tf.layers.Conv2D(
        6, (3, 3),activation='relu',
        padding='valid',name='conv9-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
        bias_initializer=biases.pop())(features)
    
    # Pad so we don't lose frequencies
    conv1 = tf.pad(conv1, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]))
    conv2 = tf.layers.Conv2D(
        6, (3, 3),activation='relu',
        padding='valid',name='conv10-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
        bias_initializer=biases.pop())(conv1)
        
    feature_map = conv2
    
    # If valid weights were loaded
    if restore:
        # Don't update layers
        tf.stop_gradient(conv1)
        tf.stop_gradient(conv2)
    
    return feature_map
    
    
# Create vector of frequencies where feedback is present
def fb_vectorize(features, kernels, biases):

    # Check for valid weights
    restore = False
    if kernels[0] is not None:
        restore = True

    _, height, width, depth = features.get_shape()
    
    # Pad so we don't lose frequencies
    features = tf.pad(features, tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]]))
    # Convolve along each frequency
    feedback_vector = tf.layers.Conv2D(
        6, (3, width),activation='relu',
        padding='valid',name='conv_vector-',
        kernel_initializer=kernels.pop(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA),
        bias_initializer=biases.pop())(features)
    
    # If valid weights were loaded
    if restore:
        # Don't update layers
        tf.stop_gradient(feedback_vector)
    
    return feedback_vector
    
# Standard 3x3 encoder
def encode3x3(features, kernels, biases):

    # Check for valid weights
    restore = False
    if kernels[0] is not None:
        restore = True

    conv1_1 = tf.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1-', kernel_initializer=kernels.pop(), bias_initializer=biases.pop())(features)
    pool1 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='pool2-')(conv1_1)
    
    conv1_2 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3-', kernel_initializer=kernels.pop(), bias_initializer=biases.pop())(pool1)
    pool2 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='pool4-')(conv1_2)
    
    conv1_3 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv5-', kernel_initializer=kernels.pop(), bias_initializer=biases.pop())(pool2)
    h = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='feature_map6-')(conv1_3)
    
    # If valid weights were loaded
    if restore:
        # Don't update layers
        tf.stop_gradient(conv1_1)
        tf.stop_gradient(conv1_2)
        tf.stop_gradient(conv1_3)
    
    return h


def decode3x3(feature_map):

    # Dense layer to get input to correct size
    _, height, width, depth = feature_map.get_shape()
    elements = int( height*width*depth )
    input_depth = math.ceil( elements/(14*14) )
    dense_elements = input_depth * 14*14
    
    feature_flat = tf.reshape(feature_map, [-1, elements])
    input = tf.layers.dense(inputs=feature_flat, units=dense_elements)
    input = tf.reshape(input, [-1, 14, 14, input_depth])

    # Decoder
    conv2_1 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv2_1)
    conv2_2 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D((2, 2))(conv2_2)
    conv2_3 = tf.layers.Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = tf.image.resize_nearest_neighbor(conv2_3, [HEIGHT, WIDTH]) # Rplaced UpSamplig2D b/c bug
    reconstructed = tf.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same',
                                     name='reconstructed_image')(up3)
                                     
    return reconstructed

def autoencoder(features, labels, mode, params):

    print("Mode:", mode)
    
    # Input Layer
    input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, 1], name="image_input")
    
    # Add noise
    noisy_layer = None
    if "noise" in params.keys():
        noisy_layer = gaussian_noise_layer(input_layer, NOISE_STD)
    
    # Autoencoders don't need pre-loaded weights
    weights = None
    
    # Choose encoder/feature_extractor
    encoder = params['feature_extractor']
    
    # Encode
    if noisy_layer is not None:
        feature_map = encode(noisy_layer, encoder)
        feature_map = encode(feature_map, feature_encoder)
    else: 
        feature_map = encode(input_layer, encoder)
        feature_map = encode(feature_map, feature_encoder)
    
    # Print dimensionality of feature map
    _, height, width, depth = feature_map.get_shape()
    print("CNN with final feature maps:", height, "x", width, "x", depth)
    print(height*width*depth, "total features")
    
    # Decode
    decode = decode3x3
    reconstructed = decode(feature_map)
    
    # Calculate Loss
    loss = tf.losses.mean_squared_error(labels=input_layer,
                                      predictions=reconstructed)
                                   
    # Put images in tensorboard
    tf.summary.image(
        "original",
        input_layer,
        max_outputs=9
      )
    if noisy_layer is not None:
        tf.summary.image(
            "noisy",
            noisy_layer,
            max_outputs=9
          )
    tf.summary.image(
        "reconstructed",
        reconstructed,
        max_outputs=9
      )
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # reducing e to 0.0001 midway through training increases accuracy
        optimizer = tf.train.AdamOptimizer(epsilon=0.00001)
        train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    
    # EVAL stuff
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "RMSE": tf.metrics.root_mean_squared_error(
          labels=input_layer, predictions=reconstructed)
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
