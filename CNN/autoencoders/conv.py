import tensorflow as tf
HEIGHT = 112
WIDTH = 112
NOIS_MEAN = 0.0
NOISE_STD = 0.2

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=NOIS_MEAN, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def encode(features, weights, encoder):

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

def conv_audio(features, kernels, biases):

    return None

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
    
def decode(feature_map):

    # Decoder
    conv2_1 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(feature_map)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv2_1)
    conv2_2 = tf.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D((2, 2))(conv2_2)
    conv2_3 = tf.layers.Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = tf.image.resize_nearest_neighbor(conv2_3, [HEIGHT, WIDTH]) # Rplaced UpSamplig2D b/c bug
    reconstructed = tf.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', 
                                     activity_regularizer=tf.nn.l2_loss,
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
    
    weights = None
    
    # Choose encoder/feature_extractor
    encoder = params['feature_extractor']
    
    # Encode
    if noisy_layer is not None:
        feature_map = encode(noisy_layer, weights, encoder)
    else: 
        feature_map = encode(input_layer, weights, encoder)
    
    # Print dimensionality of feature map
    _, height, width, depth = feature_map.get_shape()
    print("CNN with final feature maps:", height, "x", width, "x", depth)
    print(height*width*depth, "total features")
    
    # Decode
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
