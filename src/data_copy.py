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
#from progressbar import ETA, Bar, Percentage, ProgressBar

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

directory_generate_data = '../data_128/' #'../data/' #
if not os.path.exists(directory_generate_data):
    os.makedirs(directory_generate_data)

data_directory = os.path.join(FLAGS.working_directory, "MNIST")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
mnist_all = input_data.read_data_sets(data_directory, one_hot=False, validation_size = 0)

mnist_train_images = mnist_all.train.images
mnist_train_labels = mnist_all.train.labels
mnist_test_images = mnist_all.test.images
mnist_test_labels = mnist_all.test.labels

refined_label = [0, 2, 4, 6, 8]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_train_per_label = [128]


#gener_image = np.array([], dtype = np.float32)
#gener_label = np.array([], dtype = np.uint8) 
for idx, label_value in enumerate(refined_label):    
    refined_one_label_idx = np.where( mnist_train_labels == label_value )[0][:num_train_per_label[0]]
    train_refined_images = mnist_train_images[refined_one_label_idx, :]
    train_refined_labels = mnist_train_labels[refined_one_label_idx]
#import matplotlib.pyplot as plt
#
#plt.imshow(np.reshape(gener_image_per_label[500,:], (28, 28)),)    

    gener_image_per_label = np.tile( train_refined_images, (int(FLAGS.generate_size/num_train_per_label[0]), 1) )
    gener_labels_per_label = label_value * np.ones((gener_image_per_label.shape[0]), dtype=np.uint8)
    if idx == 0:
        gener_image = gener_image_per_label[:]
        gener_label = gener_labels_per_label[:]
    else:
        gener_image = np.concatenate( (gener_image, gener_image_per_label), axis=0)
        gener_label = np.concatenate( (gener_label, gener_labels_per_label), axis=0 )
    #gener_image = np.append(gener_image, gener_image_per_label, axis=0)
    #gener_label = np.append(gener_label, train_refined_labels, axis=0)

f = h5py.File(os.path.join(directory_generate_data, 'copy_data.h5'), "w")
f.create_dataset("copy_images", dtype='float32', data=gener_image)
f.create_dataset("copy_labels", dtype='uint8', data=gener_label)
f.close()

#hh = refined_label*np.ones((gener_image.shape[0]), dtype=np.uint8)


