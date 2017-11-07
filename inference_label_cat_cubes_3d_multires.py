#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:36:29 2017

@author: derek
"""

import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import sys

TARGET_VOXEL_MM = 0.682


def _parse_function(example_proto):
    """Reads tfrecords with features {shape: (height,width,depth) of cube data,
    label: (malignancy, lobulation, spiculation) labels, cube: usually 32x32x32 data). 
    Mapped onto a TFRecord dataset
    Args:
        example_proto: TFRecord protobuffer of data 
    Returns:
        shape_int32: (int32) (height,width,depth)
        label_int32: (int32) (malignancy, lobulation, spiculation)
        cube: (float32) height x width x depth data (usually 32x32x32)
    """
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

def augment_data(transpose_index,k_value,flip_yes_no, cubes):
    """augment data (cubes) by rotating the cubes k_values times, and tranposing
    the indices specified by transpose_index. 
    To randomize input: 
        transpose_index: random permutation of [0,1,2]
        k_value: random int [0-3]
        flip_yes_no: random int [0-1]
    Args:
        transpose_index: (np array) array discribing the new order of the transposed
            axis [x_axis, y_axis, z_axis] [0,1,2]-> would keep axis unchanged.
        k_value: (int) number of rotations, 0 would keep data unrotated
    Returns:
        shape_int32: (int32) (height,width,depth)
        label_int32: (int32) (malignancy, lobulation, spiculation)
        cube: (float32) height x width x depth data (usually 32x32x32)
    """
    cubes_trans = tf.map_fn(lambda img: tf.transpose(img, transpose_index), cubes)
    cubes_90 = tf.map_fn(lambda img: tf.image.rot90(img,k=k_value), cubes_trans)
    cubes_out = cubes_90
    if flip_yes_no == 1:
        cubes_out = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cubes_out)
    return cubes_out

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


def _randomize(image):
    """Add randomization to the image by raising the image values to a random 
    power between 1 and 10. Then renormalize to -.5 to .5 
    Args:
        image: input 3d data cube
    Returns:
        image: image after power and renormalized
    """
    image = image - tf.reduce_min(image)
    image = tf.pow(image, tf.random_uniform([1],minval=1,maxval=10))
    image = image/tf.reduce_max(image)
    image = image - 0.5
    return image


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
def extract_cube(image_array,z_perc,y_perc,x_perc):
    """extract a 32x32x32 chunk from data specified by the center in percentage
    (z_perc,y_perc, x_perc)
    Args:
        image_array: full size image data cube
        z_perc: the z dimensional center given as a percentage of the total z
        y_perc: the y dimensional center given as a percentage of the total y
        x_perc: the x dimensional center given as a percentage of the total x
    Returns:
        image_cube: 32x32x32 subsection of image_arrary centered at (z,y,x)
    """
    im_z, im_y, im_x = image_array.shape
    z_min = int(round(z_perc*im_z)) - 16
    y_min = int(round(y_perc*im_y)) - 16
    x_min = int(round(x_perc*im_x)) - 16
    z_max = int(round(z_perc*im_z)) + 16
    y_max = int(round(y_perc*im_y)) + 16
    x_max = int(round(x_perc*im_x)) + 16
    if z_min < 0:
        z_max = z_max + abs(z_min)
        z_min = 0
    if y_min < 0:
        y_max = y_max + abs(y_min)
        y_min = 0
    if x_min < 0:
        x_max = x_max + abs(x_min)
        x_min = 0
    if z_max > im_z:
        z_min = z_min - abs(z_max - im_z)
        z_max = im_z
    if y_max > im_y:
        y_min = y_min - abs(y_max - im_y)
        y_max = im_y
    if x_max > im_x:
        x_min = x_min - abs(x_max - im_x)
        x_max = im_x
    image_cube = image_array[z_min:z_max,y_min:y_max,x_min:x_max]
    return image_cube

def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm, verbose=False):
    """rescale patient images (3d cube data) to target_voxel_mm
    Args:
        images_zyx: full size image data cube
        org_spacing_xyz: original spacing
        target_voxel_mm: size of rescaled voxels
        verbose: print extra info
    Returns:
        image_cube: 32x32x32 subsection of image_arrary centered at (z,y,x)
    """
    if verbose:
        print("Spacing: ", org_spacing_xyz)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    interpolation = cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = np.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res

def main():
    #INPUT ARGUMENTS#
    #patient_path = "/media/derek/disk1/kaggle_ndsb2017/resources/_luna16_mhd/1.3.6.1.4.1.14519.5.2.1.6279.6001.131939324905446238286154504249.mhd"
    #model_path = "/media/derek/disk1/kaggle_ndsb2017/saved_models_mres1_50perc/model.ckpt"
    #z_perc = .2
    #y_perc = .3
    #x_perc = .4
    #################
    if len(sys.argv) != 6:
        print("Script Requires 5 arguments: model_path, patient_dicom_path, z_perc, y_perc, x_perc")
        return
    patient_path = sys.argv[1]
    model_path = sys.argv[2]
    z_perc = float(sys.argv[3])
    y_perc = float(sys.argv[4])
    x_perc = float(sys.argv[5])
    
    
    BATCH_SIZE = 1
    NUM_CLASSES = 6
    #INPUT x_perc,y_perc,z_perc, patient_path
    itk_img = SimpleITK.ReadImage(patient_path)
    if (np.array(itk_img.GetDirection()) != np.array([ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.])).all():
        print("WARNING!!!!! Image in different direction")
    image_array = SimpleITK.GetArrayFromImage(itk_img)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    image_array = rescale_patient_images(image_array, spacing, TARGET_VOXEL_MM)
    image_cube = extract_cube(image_array,z_perc,y_perc,x_perc)
    
    cubes = tf.cast(tf.constant(image_cube), tf.float32)
    
    transpose_index = tf.Variable(initial_value=[0,1,2],trainable=False,dtype=tf.int32)
    k_value = tf.Variable(initial_value=0,trainable=False,dtype=tf.int32)
    
    cubes = _normalize(cubes)  # Normalize t0 -.5 to .5.
    cubes = _randomize(cubes)
    cubes_aug = tf.reshape(cubes,[-1,32,32,32,1])
    #mal, lob, spic = tf.unstack(label,num = 3)
    
    #### CONV 1 #####
    #with tf.variable_scope('conv1') as scope:
    kernel1 = _variable_with_weight_decay('weights1',
                                         shape=[3, 3, 3, 1, 8],
                                         stddev=5e-2,
                                         wd=0.0)
    conv1_ = tf.nn.conv3d(cubes_aug, kernel1, [1, 1, 1, 1, 1], padding='SAME')
    biases1 = _variable_initializer('biases1', [8], tf.constant_initializer(0.0))
    pre_activation1 = tf.nn.bias_add(conv1_, biases1)
    conv1 = tf.nn.relu(pre_activation1, name='scope.name1')
    _activation_summary(conv1)
    
    # pool1
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = pool1 
    
    #### CONV 2 #####
    #with tf.variable_scope('conv2') as scope:
    avg1 = tf.nn.avg_pool3d(cubes_aug, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                         padding='SAME', name='poolavg2')
    input2 = tf.concat([norm1, avg1],axis=4)
    
    kernel2 = _variable_with_weight_decay('weights2',
                                         shape=[3, 3, 3, 9, 24],
                                         stddev=5e-2,
                                         wd=0.0)
    conv2_ = tf.nn.conv3d(input2, kernel2, [1, 1, 1, 1, 1], padding='SAME')
    biases2 = _variable_initializer('biases2', [24], tf.constant_initializer(0.1))
    pre_activation2 = tf.nn.bias_add(conv2_, biases2)
    conv2 = tf.nn.relu(pre_activation2, name='scope.name2')
    _activation_summary(conv2)
    
    # norm2
    norm2 = conv2
    
    # pool2
    pool2 = tf.nn.max_pool3d(norm2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                           padding='SAME', name='pool2')
    
    # norm2
    norm2 = pool2 
    
    #### CONV 3 #####
    avg2 = tf.nn.avg_pool3d(avg1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                         padding='SAME', name='poolavg3')
    input3 = tf.concat([norm2, avg2],axis=4)
    
    kernel3 = _variable_with_weight_decay('weights3',
                                         shape=[3, 3, 3, 25, 48],
                                         stddev=5e-2,
                                         wd=0.0)
    conv3_ = tf.nn.conv3d(input3, kernel3, [1, 1, 1, 1, 1], padding='SAME')
    biases3 = _variable_initializer('biases3', [48], tf.constant_initializer(0.1))
    pre_activation3 = tf.nn.bias_add(conv3_, biases3)
    conv3 = tf.nn.relu(pre_activation3, name='scope.name3')
    _activation_summary(conv3)
    
    # norm3
    norm3 = conv3
    #tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                name='norm2')
    # pool2
    pool3 = tf.nn.max_pool3d(norm3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                           padding='SAME', name='pool3')
    
    # norm2
    norm3 = pool3
    #### CONV 4 #####
    avg3 = tf.nn.avg_pool3d(avg2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                         padding='SAME', name='poolavg4')
    input4 = tf.concat([norm3, avg3],axis=4)
    
    kernel4 = _variable_with_weight_decay('weights4',
                                         shape=[3, 3, 3, 49, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv4_ = tf.nn.conv3d(input4, kernel4, [1, 1, 1, 1, 1], padding='SAME')
    biases4 = _variable_initializer('biases4', [64], tf.constant_initializer(0.1))
    pre_activation4 = tf.nn.bias_add(conv4_, biases4)
    conv4 = tf.nn.relu(pre_activation4, name='scope.name4')
    _activation_summary(conv4)
    
    # norm4
    norm4 = conv4
    #tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                name='norm2')
    # pool4
    pool4 = tf.nn.max_pool3d(norm4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                           padding='SAME', name='pool4')
    
    # norm4
    norm4 = pool4
    avg4 = tf.nn.avg_pool3d(avg3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                         padding='SAME', name='poolavg4')
    #################
    input5 = tf.concat([norm4, avg4],axis=4)
    ################
    
    # local3
    pool2_flatten = tf.reshape(input5, [BATCH_SIZE, -1])
    dim = 520
    weights = _variable_with_weight_decay('weights5', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_initializer('biases5', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(pool2_flatten, weights) + biases, name='scope.name5')
    _activation_summary(local3)
    
    # local4
    weights = _variable_with_weight_decay('weights6', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_initializer('biases6', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='scope.name6')
    _activation_summary(local4)
    
    # softmax linear layer (WX + b),
    weights = _variable_with_weight_decay('weights7', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_initializer('biases7', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='scope.name7')
    softmax = tf.nn.softmax(softmax_linear)
    
    
    
    
    #Start session and run inference
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, model_path)
    test_results = sess.run(softmax,feed_dict={transpose_index: [0,1,2], k_value: 0})
    print(test_results)
    
    
if __name__ == "__main__":
    main()



