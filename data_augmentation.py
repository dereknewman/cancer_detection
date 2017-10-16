#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:45:40 2017

@author: derek
"""


import os
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from matplotlib import animation

BATCH_SIZE = 128
NUM_CLASSES = 6

def _parse_function(example_proto):
  features = {"shape": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.string, default_value=""),
              "cube": tf.FixedLenFeature((), tf.string, default_value="")}
  parsed_features = tf.parse_single_example(example_proto, features)
  shape = tf.decode_raw(parsed_features['shape'], tf.int16)
  shape_int32 = tf.cast(shape,tf.int32)
  label = tf.decode_raw(parsed_features['label'], tf.int16)
  label_int32 = tf.cast(label,tf.int32)
  cube_flat = tf.decode_raw(parsed_features['cube'], tf.int16)
  cube_flat_f32 = tf.cast(cube_flat,dtype=tf.float32)
  cube = tf.reshape(cube_flat_f32,[shape_int32[0],shape_int32[1],shape_int32[2]])
  return shape_int32,label_int32,cube

def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

def _normalize(image):
    """ Normalize image -> clip data between -1000 and 400. Scale values to -0.5 to 0.5 
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = tf.maximum(MIN_BOUND, image)
    image = tf.minimum(MAX_BOUND, image)
    image = (image - MIN_BOUND)
    image = image / (MAX_BOUND - MIN_BOUND)
    image = image - 0.5
    return image

global_step = tf.contrib.framework.get_or_create_global_step()

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.shuffle(buffer_size=10000)

dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_initializable_iterator()

next_element = iterator.get_next()
shape,label,cubes = next_element
cubes = _normalize(cubes)  # Normalize t0 -.5 to .5.

transpose_index = tf.Variable(initial_value=[0,1,2],trainable=False,dtype=tf.int32)
cubes_trans = tf.map_fn(lambda img: tf.transpose(img, transpose_index), cubes)

k_value = tf.Variable(initial_value=0,trainable=False,dtype=tf.int32)
cubes_90 = tf.map_fn(lambda img: tf.image.rot90(img,k=k_value), cubes_trans)

cubes_lr = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cubes_90)

#mal, lob, spic = tf.unstack(label,num = 3)
mal, lob, spic = tf.split(label,3,axis=1)
label_onehot = tf.one_hot(mal,6)
label_f= tf.reshape(mal,[BATCH_SIZE])


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
src_dir_train = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/train/"
src_dir_test = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/test/"
filenames_train = os.listdir(src_dir_train)
filenames_test = os.listdir(src_dir_test)
training_filenames = [src_dir_train + f for f in filenames_train]
testing_filenames = [src_dir_test + f for f in filenames_test]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})


aa,bb = sess.run([cubes, cubes_lr], feed_dict={transpose_index: [0,1,2], k_value: 0})
#np.alltrue(aa[0,:,:,:]==bb[0,:,:,:])

[shape,label,cube], cubes_lr = sess.run([next_element,cubes_lr])


print(sess.run(shape))
c = sess.run(cubes)
sh = sess.run(shape)
l = sess.run(label)
m = sess.run(mal)


#############################################################################
############ ANIMATE cubes AND cubes_lr ####################################
#############################################################################
c,c_lr = sess.run([cubes, cubes_lr], feed_dict={transpose_index: [2,1,0], k_value: 2})
image = c[0,:,:,:]
image2 = c_lr[0,:,:,:]

#setup figure
fig = plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

#set up list of images for animation
ims=[]
for time in range(np.shape(image)[1]):
    im = ax1.imshow(image[time,:,:],cmap="gray")
    im2 = ax2.imshow(image2[time,:,:],cmap="gray")
    ims.append([im, im2])

#run animation
ani = animation.ArtistAnimation(fig,ims, interval=50,blit=False)
plt.show()
#############################################################################
#############################################################################







