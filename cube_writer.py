#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:22:17 2017

@author: derek
"""

import numpy as np
import tensorflow as tf



def get_image_binary(filename):
    image_cube = np.load(filename)
    image_cube = np.asarray(image_cube,np.int16)
    shape = np.array(image_cube.shape, np.int32)
    return shape.tobytes(), image_cube.tobytes() #convert image to raw data bytes in the array



def write_to_tfrecord(labels, shape, binary_image, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image]))
            }))
    writer.write(example.SerializeToString())
    writer.close()
    
    
def read_from_tfrecord(filename):
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    filename = "cubes1.tfrecord"
    reader = tf.TFRecordReader()
    key, tfrecord_serialized = reader.read(filename)
    # Convert from a string to a vector of uint8 that is record_bytes long.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,feautres={
                  'label' : tf.FixedLenFeature([], tf.string),
                  'shape' : tf.FixedLenFeature([], tf.string),
                  'image' : tf.FixedLenFeature([], tf.string),
                  }, name='features')
    image = tf.decode_raw(tfrecord_features['image'],tf.int16)
    shape = tf.decode_raw(tfrecord_features['shape'],tf.int32)
    label = tf.decode_raw(tfrecord_features['label'],tf.int16)
    image_cube = tf.reshape(image, shape)
    return label, shape, image_cube


#$$#$#
def patient_to_tfrecord(patient_id, image_array, patient_df):
    patient_id = "1.4.5.6.123551485448654"
    tfrecord_file = patient_id + ".tfrecord"  
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(1000):
        image_cube = np.random.randint(-1000,1000,[32,32,32],dtype=np.int16)
        image_label = np.random.randint(0,5,3,dtype=np.int16)
        
        
        
        image_cube = np.asarray(image_cube,np.int16) #ensure data is in int16
        binary_cube = image_cube.tobytes()
        image_label = np.array(image_label,np.int16) #ensure data is in int16
        binary_label = image_label.tobytes()
        shape = np.array(image_cube.shape, np.int32) #ensure data is in int16
        binary_shape = shape.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_label])),
                'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_shape])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_cube]))
                }))
        writer.write(example.SerializeToString())
    writer.close()

patient_id = "1.4.5.6.123551485448654"
tfrecord_file = patient_id + ".tfrecord"  
writer = tf.python_io.TFRecordWriter(tfrecord_file)
for i in range(1000):
    image_cube = np.random.randint(-1000,1000,[32,32,32],dtype=np.int16)
    image_label = np.random.randint(0,5,3,dtype=np.int16)
    
    
    
    image_cube = np.asarray(image_cube,np.int16) #ensure data is in int16
    binary_cube = image_cube.tobytes()
    image_label = np.array(image_label,np.int16) #ensure data is in int16
    binary_label = image_label.tobytes()
    shape = np.array(image_cube.shape, np.int32) #ensure data is in int16
    binary_shape = shape.tobytes()
    
    example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_label])),
            'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_shape])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_cube]))
            }))
    writer.write(example.SerializeToString())
writer.close()


#$#$#$#
filenames = ["cubes1.tfrecord"]
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






