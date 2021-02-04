# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:48:37 2020

@author: Jason
"""

# EEEN4/60151 Machine Learning Laboratory
# 1. Single Perceptron


import numpy as np
from Model_class import Model
from ImageData_class import ImageData

# Create the 0 and 1 images as numpy array and the corresponding labels
ones = np.array([[[0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0]], 
                 [[0,1,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,1,1,1,0]],
                 [[0,0,1,0,0],
                  [0,1,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0]],
                 [[0,0,0,1,0],
                  [0,0,1,1,0],
                  [0,0,1,0,0],
                  [0,1,1,0,0],
                  [0,1,0,0,0]],
                 [[0,0,0,0,1],
                  [0,0,0,1,0],
                  [0,0,1,0,0],
                  [0,1,0,0,0],
                  [1,0,0,0,0]], 
                 [[0,0,0,0,1],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0]]
                 ])

zeros = np.array([[[0,1,1,1,0],
                   [0,1,0,1,0],
                   [0,1,0,1,0],
                   [0,1,0,1,0],
                   [0,1,1,1,0]],
                  [[0,0,1,0,0],
                   [0,1,0,1,0],
                   [0,1,0,1,0],
                   [0,1,0,1,0],
                   [0,0,1,0,0]],
                  [[0,0,1,0,0],
                   [0,1,0,1,0],
                   [0,1,0,1,0],
                   [0,1,0,1,0],
                   [0,1,1,1,0]],
                  [[0,0,1,1,0],
                   [0,1,0,0,1],
                   [0,1,0,0,1],
                   [0,1,0,1,0],
                   [0,1,1,0,0]],
                  [[1,1,1,1,1],
                   [1,0,0,0,1],
                   [1,0,1,0,1],
                   [1,0,0,0,1],
                   [1,1,1,1,1]],
                  [[0,1,1,1,0],
                   [1,0,0,0,1],
                   [1,0,0,0,1],
                   [1,0,0,0,1],
                   [0,1,1,1,0]]
                  ])

one_labels = np.array([[1],
                       [1],
                       [1],
                       [1],
                       [1], 
                       [1]
                      ])

zero_labels = np.array([[-1],
                        [-1],
                        [-1],
                        [-1],
                        [-1],
                        [-1]
                       ])

# Preprocess the images and labels using the ImageData class
# For this task the image will need to be in the format 
# (image, example) with flattened images
# The training set will have 8 examples (4 1s, 4 0s) and the
# validation set will have 4 examples (2 1s, 2 0s)
data = ImageData()
data.get_data(images1=ones, 
              image_labels1=one_labels, 
              images2=zeros, 
              image_labels2=zero_labels
              )
data.data_preprocess(split_data=True,
                     normalize=False,
                     flatten=True,
                     img_first_format=True, 
                     one_hot_encode=False
                     )

# Define and train a single perceptron model for learning this dataset, then
# plot the training and validation accuracy as a function of epoch and save
# this data to a csv file
input_size = data.train_data.shape[0]
output_size = data.train_labels.shape[1]
save_dir = "C:/Users/Jason/Documents/"
graph_save_path = save_dir + "graph.png"
data_save_path = save_dir + "data.csv"

single_layer_perceptrons = Model(input_size,
                                 output_size,
                                 hidden_size=None,
                                 output_activation="sign"
                                 )
single_layer_perceptrons.train(data.train_data,
                               data.train_labels,
                               data.val_data,
                               data.val_labels,
                               init_factor=0.1,
                               lrn_rate=1e-2,
                               max_accept_error=0
                               )
single_layer_perceptrons.plot(graph_save_path)
single_layer_perceptrons.save(data_save_path)