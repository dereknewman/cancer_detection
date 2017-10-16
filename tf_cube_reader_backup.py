#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:36:29 2017

@author: derek
"""

import os
import tensorflow as tf
import numpy as np

# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
#def parser(record):
#    keys_to_features = {
#        "label": tf.FixedLenFeature((), tf.string, default_value=""),
#        "shape": tf.FixedLenFeature((), tf.string, default_value=""),
#        "image": tf.FixedLenFeature((), tf.string, default_value="")
#    }
#    keys_to_features = {
#    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
#                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
#                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_depth])),
#                'mal': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_mal])),
#                'spic': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_spic])),
#                'lob': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_lob])),
#                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_cube]))
#    parsed = tf.parse_single_example(record, keys_to_features)
#
#    # Perform additional preprocessing on the parsed data.
#    image = tf.decode_jpeg(parsed["image_data"])
#    image = tf.reshape(image, [299, 299, 1])
#    label = tf.cast(parsed["label"], tf.int32)
#
#    return {"image_data": image, "date_time": parsed["date_time"]}, label

# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"shape": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.string, default_value=""),
              "cube": tf.FixedLenFeature((), tf.string, default_value="")}
  parsed_features = tf.parse_single_example(example_proto, features)
  shape = tf.decode_raw(parsed_features['shape'], tf.int16)
  label = tf.decode_raw(parsed_features['label'], tf.int16)
  cube = tf.decode_raw(parsed_features['cube'], tf.int16)
  return shape,label,cube

def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

#src_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/"
#filenames = os.listdir(src_dir)
#filenames = [src_dir + f for f in filenames]
#dataset = tf.contrib.data.TFRecordDataset(filenames)
#dataset = dataset.map(_parse_function)  # Parse the record into tensors.
##dataset = dataset.repeat()  # Repeat the input indefinitely.
##dataset = dataset.batch(32)
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#sess = tf.Session()
#
#shape,label,cube = sess.run(next_element)
#cube = np.reshape(cube,shape)

##########################################################################
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

next_element = iterator.get_next()
shape,label,cube = next_element
sess = tf.Session()
src_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/"
filenames_all = os.listdir(src_dir)
training_filenames = [src_dir + f for f in filenames_all]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

print(sess.run(shape))


validation_filenames = [src_dir + f for f in filenames_all]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

shape,label,cube = sess.run(next_element)
cube = np.reshape(cube,shape)






sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break




sess = tf.Session()
# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]






#TEST
#dataset = tf.contrib.data.Dataset.range(100)
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
sess = tf.Session()
sess.run(iterator.initializer)
next_element = iterator.get_next()
print(sess.run(next_element))






while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break




tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
reader = tf.TFRecordReader()
key, tfrecord_serialized = reader.read(tfrecord_file_queue)
# Convert from a string to a vector of uint8 that is record_bytes long.
tfrecord_features = tf.parse_single_example(tfrecord_serialized,features={
              'label' : tf.FixedLenFeature([], tf.string),
              'shape' : tf.FixedLenFeature([], tf.string),
              'image' : tf.FixedLenFeature([], tf.string),
              }, name='features')
image = tf.decode_raw(tfrecord_features['image'],tf.int16)
shape = tf.decode_raw(tfrecord_features['shape'],tf.int32)
label = tf.decode_raw(tfrecord_features['label'],tf.int16)
image_cube = tf.reshape(image, shape)

##########################################################################

#filenames = ["cubes1.tfrecord", "cubes2.tfrecord"]
#dataset = tf.contrib.data.TFRecordDataset(filenames)

# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)





print(sess.run(label))



