import tensorflow as tf
import numpy as np
import matplotlib;
import cv2

def parse_record(serialized_example): #parse a single binary example
  """Parses a single tf.Example into image and label tensors."""
  features = {'image/encoded': tf.FixedLenFeature([], tf.string),
             'image/format':  tf.FixedLenFeature([], tf.string),
             'image/label':   tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(serialized_example, features)
  
  #print("JPG:", features['image/encoded'])
  image = tf.image.decode_jpeg(features['image/encoded'], channels=0)
  #print("image:", image)
  image = tf.reshape(image, [40, 398, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  
  label = tf.cast(features['image/label'], tf.int32)
  
  return (image, label)

train_filenames = ["nsynth/test.record"]
batch_size=256
dataset = tf.data.TFRecordDataset(train_filenames)
dataset = tf.data.TFRecordDataset(train_filenames)
dataset = dataset.map(parse_record)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
features, labels = iterator.get_next()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  
  images = [0]
  while len(images) > 0:
    images = features.eval()
    labs = labels.eval()
    cv2.destroyAllWindows()
    count = 0
  
    for image in images:
      count = count + 1
      if not image.any():
        cv2.imshow("bad image", np.asarray(image))
        cv2.waitKey(0) #wait for user input
      else:
        cv2.imshow("image", np.asarray(image))
        cv2.waitKey(10)
