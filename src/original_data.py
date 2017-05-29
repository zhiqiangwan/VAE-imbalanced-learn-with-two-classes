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
#from progressbar import ETA, Bar, Percentage, ProgressBar


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")
flags.DEFINE_string("generate_size", 5600, "batch size of generated images")

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

refined_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_minority_label = 128
num_majority_label = 5000
num_train_per_label = [num_minority_label, num_majority_label]*5
train_refined_label_idx = np.array([], dtype = np.uint8)
test_refined_label_idx = np.array([], dtype = np.uint8)
for idx, label_value in enumerate(refined_label):
    refined_one_label_idx = np.where( mnist_train_labels == label_value )[0][:num_train_per_label[idx]]
    train_refined_label_idx = np.append( train_refined_label_idx,  refined_one_label_idx)
    test_refined_one_label_idx = np.where( mnist_test_labels == label_value )[0]
    test_refined_label_idx = np.append(test_refined_label_idx, test_refined_one_label_idx)

#odd_labels = [1, 3, 5, 7, 9]
#even_labels = [0, 2, 4, 6, 8]
#num_train_per_label = [400, 5000] #odd labels are minority
##two_class_labels = [0, 1]
#
#odd_label_idx = np.array([], dtype = np.uint8)
#odd_test_label_idx = np.array([], dtype = np.uint8)
#for idx, label_value in enumerate(odd_labels):
#    refined_one_label_idx = np.where( mnist_train_labels == label_value )[0][:num_train_per_label[0]]
#    odd_label_idx = np.append(odd_label_idx, refined_one_label_idx)
#    test_refined_one_label_idx = np.where( mnist_test_labels == label_value )[0]
#    odd_test_label_idx = np.append(odd_test_label_idx, test_refined_one_label_idx)
#
#odd_refined_images = mnist_train_images[odd_label_idx, :]
##odd_refined_labels = two_class_labels[0]*np.ones((odd_refined_images.shape[0]), dtype=np.uint8)
#odd_test_refined_images = mnist_test_images[odd_test_label_idx, :]
##odd_test_refined_labels = two_class_labels[0]*np.ones((odd_test_refined_images.shape[0]), dtype=np.uint8)


train_refined_images = mnist_train_images[train_refined_label_idx, :]
train_refined_labels = mnist_train_labels[train_refined_label_idx]
test_refined_images = mnist_test_images[test_refined_label_idx, :]
test_refined_labels = mnist_test_labels[test_refined_label_idx]

f = h5py.File(os.path.join(directory_generate_data, 'original_data.h5'), "w")
f.create_dataset("train_refined_images", dtype='float32', data=train_refined_images)
f.create_dataset("train_refined_labels", dtype='uint8', data=train_refined_labels)
f.create_dataset("test_refined_images", dtype='float32', data=test_refined_images)
f.create_dataset("test_refined_labels", dtype='uint8', data=test_refined_labels)
f.close()

#
#
#plt.imshow(np.reshape(train_refined_images[500,:], (28, 28)),)

