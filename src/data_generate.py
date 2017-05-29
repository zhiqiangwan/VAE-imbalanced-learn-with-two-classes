'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import h5py
import matplotlib.pyplot as plt

from vae import VAE
from gan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")
flags.DEFINE_string("generate_size", 4600, "batch size of generated images")

FLAGS = flags.FLAGS

directory_generate_data = '../data_384/' #'../data_128/' #
if not os.path.exists(directory_generate_data):
    os.makedirs(directory_generate_data)

#input data processing
data_directory = os.path.join(FLAGS.working_directory, "MNIST")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
mnist_all = input_data.read_data_sets(data_directory, one_hot=False, validation_size = 0)

mnist_train_images = mnist_all.train.images
mnist_train_labels = mnist_all.train.labels
mnist_test_images = mnist_all.test.images
mnist_test_labels = mnist_all.test.labels

num_minority_label = 384
even_labels = [0, 2, 4, 6, 8]
two_class_labels = [0, 1]

even_label_idx = np.array([], dtype = np.uint8)
even_test_label_idx = np.array([], dtype = np.uint8)
for idx, label_value in enumerate(even_labels):
    refined_one_label_idx = np.where( mnist_train_labels == label_value )[0][:num_minority_label]
    even_label_idx = np.append(even_label_idx, refined_one_label_idx)

even_refined_images = mnist_train_images[even_label_idx, :]
even_refined_labels = two_class_labels[1]*np.ones((even_refined_images.shape[0]), dtype=np.uint8)
    

#gener_image = np.array([], dtype = np.float32)
#gener_label = np.array([], dtype = np.uint8) 

#
#plt.imshow(np.reshape(even_refined_images[35+384*0,:], (28, 28)),)

assert FLAGS.model in ['vae', 'gan']
if FLAGS.model == 'vae':
    model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.generate_size)
elif FLAGS.model == 'gan':
    model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.generate_size)
   
num_train = even_refined_images.shape[0]
loss_list = []
iterations = 0

for epoch in range(FLAGS.max_epoch):

    trainset_shuffle_index = np.random.permutation(num_train)
        
    for start, end in zip( range( 0, num_train+FLAGS.batch_size, FLAGS.batch_size ),
                          range( FLAGS.batch_size, num_train+FLAGS.batch_size, FLAGS.batch_size ) ):
       if end > num_train: end = num_train
       
       images = even_refined_images[trainset_shuffle_index[start:end], :]

       loss_value = model.update_params(images)
       loss_list.append(loss_value)
       
       iterations += 1
       if iterations % 500 == 0:
           print("======================================")
           print("Epoch", epoch, "Iteration", iterations) 
           
           print ("Training Loss:", np.mean(loss_list))
           print ("\n")
           loss_list = []      
    if epoch % 400 == 0 or epoch == (FLAGS.max_epoch-1):       
        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)

gener_image = model.sess.run(model.sampled_tensor_gener)
gener_label =  even_refined_labels[0] * np.ones((gener_image.shape[0]), dtype=np.uint8)

if FLAGS.model == 'vae':    
    f = h5py.File(os.path.join(directory_generate_data, 'VAE_generated_data.h5'), "w")
    f.create_dataset("VAE_images", dtype='float32', data=gener_image)
    f.create_dataset("VAE_labels", dtype='uint8', data=gener_label)
    f.close()
elif FLAGS.model == 'gan':
    f = h5py.File(os.path.join(directory_generate_data, 'GAN_generated_data.h5'), "w")
    f.create_dataset("GAN_images", dtype='float32', data=gener_image)
    f.create_dataset("GAN_labels", dtype='uint8', data=gener_label)
    f.close()    
#hh = refined_label*np.ones((gener_image.shape[0]), dtype=np.uint8)


