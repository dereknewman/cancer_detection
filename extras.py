#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:18:52 2017

@author: derek
"""

#def patient_to_tfrecord(save_path,patient_id, image_array, patient_df):
#    patient_id = "1.4.5.6.123551485448654"
#    tfrecord_file = patient_id + ".tfrecord"  
#    writer = tf.python_io.TFRecordWriter(tfrecord_file)
#    for i in range(1000):
#        add_to_tfrecord(writer,image_cube, label)
#    writer.close()

#def add_to_tfrecord(writer,image_cube, label):
#        image_cube = np.asarray(image_cube,np.int16) #ensure data is in int16
#        binary_cube = image_cube.tobytes()
#        image_mal, image_spic, image_lob = np.array(label,np.int64) #ensure data is in int16
#        image_height, image_width, image_depth = np.array(image_cube.shape, np.int64) #ensure data is in int16
#        
#        example = tf.train.Example(features=tf.train.Features(feature={
#                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
#                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
#                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_depth])),
#                'mal': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_mal])),
#                'spic': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_spic])),
#                'lob': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_lob])),
#                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_cube]))
#                }))
#        writer.write(example.SerializeToString())


import argparse



parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()
#FLAGS.use_fp16 = False
# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')


#t,l1h = sess.run([test,label_onehot_i64])
#shape,label,cube = sess.run(next_element)
#print(sess.run(shape))
#c = sess.run(cubes)
#sh = sess.run(shape)
#l = sess.run(label)
#m = sess.run(mal)


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



####TESTING DATA AUGMENTATION#####
fig = plt.figure()
ims = []
for i in range(32):
    im_slide = plt.imshow(c[:,:,i], cmap="gray",animated=True)
    ims.append([im_slide])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=0)

plt.show()

####TESTING DATA AUGMENTATION#####
im_cube = []
for ii in range(32):
    im_slice = []
    for i in range(32):
        ch1 = np.arange(0,.32,0.01)
        im_slice.append(ch1+ii)
    im_c = np.stack(im_slice)
    im_cube.append(im_c)
im_cube = np.stack(im_cube)

test_cube = tf.constant(im_cube,dtype=tf.float32)
test_cube_rot = tf.transpose(test_cube, [2, 1, 0])
test_cube_rot2 = tf.transpose(test_cube_rot, [2, 1, 0])

im_cube_r = sess.run(test_cube_rot)
im_cube_r2 = sess.run(test_cube_rot2)
plt.figure()
plt.imshow(im_cube[:,:,0])
plt.figure()
plt.imshow(im_cube_r[0,:,:])
####TESTING DATA AUGMENTATION#####
#cubes_90 = tf.image.rot90(cubes, k=1, name=None)
#cubes_90 = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cubes)

##TODO: test_cube_rot2 = tf.transpose(test_cube_rot, [2, 1, 0])
#random_condition1 = tf.random_uniform([3],minval=0,maxval=3,dtype=tf.float32)
#cubes_90_ = tf.cond(random_condition1 > 1, lambda: tf.map_fn(lambda img: tf.image.rot90(img,k=1), cubes), lambda: cubes)
#cubes_90_ = tf.cond(random_condition1 > 2, lambda: tf.map_fn(lambda img: tf.image.rot90(img,k=2), cubes), lambda: cubes)
#cubes_90_ = tf.cond(random_condition1 > 3, lambda: tf.map_fn(lambda img: tf.image.rot90(img,k=3), cubes), lambda: cubes)

#TODO: test_cube_rot2 = tf.transpose(test_cube_rot, [2, 1, 0])
#random_condition1 = tf.random_uniform([],minval=0,maxval=6,dtype=tf.int32)
#transpose_possiblities = tf.constant(np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]))

#random_condition2 = tf.random_uniform([],minval=0,maxval=4,dtype=tf.int32)
#cubes_90 = tf.cond(tf.equal(random_condition2,1), lambda: tf.map_fn(lambda img: tf.image.rot90(img,k=1), cubes_trans), lambda: cubes_trans)
#cubes_90 = tf.cond(tf.equal(random_condition2,2), lambda: tf.map_fn(lambda img: tf.image.rot90(img,k=2), cubes_trans), lambda: cubes_trans)
#cubes_90 = tf.cond(tf.equal(random_condition2,3), lambda: tf.map_fn(lambda img: tf.image.rot90(img,k=3), cubes_trans), lambda: cubes_trans)
#
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label


def plot_slice_box(image_array,z_perc,y_perc,x_perc):
    image_array = image_array - image_array.min()
    image_array = (image_array/image_array.max())*255
    z_,y_,x_ = image_array.shape
    z_loc = int(round(z_*z_perc))
    y_loc = int(round(y_*y_perc))
    x_loc = int(round(x_*x_perc))
    im_slice = image_array[z_loc,:,:].copy()
    im_slice[y_loc-16:y_loc+16,x_loc-16] = 255
    im_slice[y_loc-16:y_loc+16,x_loc+16] = 255
    im_slice[y_loc-16,x_loc-16:x_loc+16] = 255
    im_slice[y_loc+16,x_loc-16:x_loc+16] = 255
    #plt.imshow(im_slice,cmap="gray")
    cv2.imwrite(BASE_DIR + "/resources/_images/img_" + str(z_loc)+'_'+str(y_loc)+'_'+str(x_loc) + ".png", im_slice)
    
    
    
    
    
    


