#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:36:29 2017

@author: derek
"""

import os
import tensorflow as tf
import numpy as np
import argparse

#TODO:


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()
BATCH_SIZE = 128
#FLAGS.use_fp16 = False
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

global_step = tf.contrib.framework.get_or_create_global_step()

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_initializable_iterator()

next_element = iterator.get_next()
shape,label,cubes = next_element

#mal, lob, spic = tf.unstack(label,num = 3)
mal, lob, spic = tf.split(label,3,axis=1)
label_onehot = tf.one_hot(mal,6)
label_f= tf.reshape(mal,[BATCH_SIZE])





##########################################################################
##########################################################################
def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
    decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    var = _variable_initializer(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_initializer(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #tf.summary.histogram(tensor_name + '/activations', x)
    #tf.summary.scalar(tensor_name + '/sparsity',
    #tf.nn.zero_fraction(x))
    #tf.summary.histogram(x)
    #tf.summary.scalar(x)
    pass

"""Build the CIFAR-10 model.
Args:
images: Images returned from distorted_inputs() or inputs().
Returns:
Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
#with tf.variable_scope('conv1') as scope:
kernel1 = _variable_with_weight_decay('weights1',
                                     shape=[5, 5, 32, 64],
                                     stddev=5e-2,
                                     wd=0.0)
conv1_ = tf.nn.conv2d(cubes, kernel1, [1, 1, 1, 1], padding='SAME')
biases1 = _variable_initializer('biases1', [64], tf.constant_initializer(0.0))
pre_activation1 = tf.nn.bias_add(conv1_, biases1)
conv1 = tf.nn.relu(pre_activation1, name='scope.name1')
_activation_summary(conv1)

# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                     padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm1')

# conv2
#with tf.variable_scope('conv2') as scope:
kernel2 = _variable_with_weight_decay('weights2',
                                     shape=[5, 5, 64, 64],
                                     stddev=5e-2,
                                     wd=0.0)
conv2_ = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
biases2 = _variable_initializer('biases2', [64], tf.constant_initializer(0.1))
pre_activation2 = tf.nn.bias_add(conv2_, biases2)
conv2 = tf.nn.relu(pre_activation2, name='scope.name2')
_activation_summary(conv2)

# norm2
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm2')
# pool2
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                     strides=[1, 2, 2, 1], padding='SAME', name='pool2')






# local3
#with tf.variable_scope('local3') as scope:
# Move everything into depth so we can perform a single matrix multiply.
pool2_flatten = tf.reshape(pool2, [BATCH_SIZE, -1])
dim = 4096
weights = _variable_with_weight_decay('weights3', shape=[dim, 384],
                                      stddev=0.04, wd=0.004)
biases = _variable_initializer('biases3', [384], tf.constant_initializer(0.1))
local3 = tf.nn.relu(tf.matmul(pool2_flatten, weights) + biases, name='scope.name3')
_activation_summary(local3)

# local4
#with tf.variable_scope('local4') as scope:
weights = _variable_with_weight_decay('weights4', shape=[384, 192],
                                      stddev=0.04, wd=0.004)
biases = _variable_initializer('biases4', [192], tf.constant_initializer(0.1))
local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='scope.name4')
_activation_summary(local4)

# linear layer(WX + b),
# We don't apply softmax here because
# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
# and performs the softmax internally for efficiency.
#with tf.variable_scope('softmax_linear') as scope:
weights = _variable_with_weight_decay('weights5', [192, NUM_CLASSES],
                                      stddev=1/192.0, wd=0.0)
biases = _variable_initializer('biases5', [NUM_CLASSES],
                          tf.constant_initializer(0.0))
softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='scope.name')
_activation_summary(softmax_linear)

#return softmax_linear
##########################################################################
##########################################################################
"""Add L2Loss to all the trainable variables.
Add summary for "Loss" and "Loss/avg".
Args:
logits: Logits from inference().
labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
Returns:
Loss tensor of type float.
"""
# Calculate the average cross entropy loss across the batch.
#label_onehot_i64 = tf.cast(label_onehot, tf.int64)
label_onehot_= tf.reshape(label_onehot,[128,6])
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_f, logits=softmax_linear, name='cross_entropy_per_example')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#tf.add_to_collection('losses', cross_entropy_mean)

# The total loss is defined as the cross entropy loss plus all of the weight
# decay terms (L2 loss).
#return tf.add_n(tf.get_collection('losses'), name='total_loss')
##########################################################################
##########################################################################
lr = 0.003
optimizer_ = tf.train.GradientDescentOptimizer(lr)
grads = optimizer_.compute_gradients(cross_entropy_mean)
# Apply gradients.
apply_gradient_op = optimizer_.apply_gradients(grads, global_step=global_step)
train_op = apply_gradient_op


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
src_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_tfrecords/"
filenames_all = os.listdir(src_dir)
training_filenames = [src_dir + f for f in filenames_all]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

print(sess.run(shape))
c = sess.run(cubes)
sh = sess.run(shape)
l = sess.run(label)
m = sess.run(mal)

#t,l1h = sess.run([test,label_onehot_i64])

validation_filenames = [src_dir + f for f in filenames_all]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

shape,label,cube = sess.run(next_element)


for index in range(100000):
    if (index % 100)==0:
        print(sess.run(cross_entropy_mean))
    sess.run(train_op)





#
#sess = tf.Session()
## Compute for 100 epochs.
#for _ in range(100):
#  sess.run(iterator.initializer)
#  while True:
#    try:
#      sess.run(next_element)
#    except tf.errors.OutOfRangeError:
#      break
#
#  # [Perform end-of-epoch calculations here.]





