# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:00:48 2020

@author: Jason
"""

# 2. Single Layer Perceptrons

import mnist
from ImageData import ImageData
from Model import Model

# Preprocess the images and labels using the ImageData class
# For this task the image will need to be in the format 
# (image, example) with flattened images and one-hot-encoded
# labels (of size (n_examples, 10) as there are 10 possible labels 0-9)
# MNIST has 60,000 training examples and 10,000 validation examples
data = ImageData()
data.get_data(dataset=mnist)
data.data_preprocess(pre_shuffle=False,
                     normalize=True,
                     flatten=True,
                     img_first_format=True,
                     one_hot_encode=True
                     )

# Define and use a single layer of perceptrons model for learning the MNIST
# data, then plot the training and validation accuracy as a function of epoch
# and save this data to a csv file
input_size = data.train_data.shape[0]
output_size = data.train_labels.shape[1]
save_dir = "C:/Users/Jason/Documents/"
save_name = "save_name"
graph_save_path = save_dir + save_name + "-graph.png"
data_save_path = save_dir + save_name + "-data.csv"

single_layer_perceptrons = Model(input_size,
                                 output_size,
                                 hidden_size=None,
                                 output_activation="sigmoid"
                                 )
single_layer_perceptrons.train(data.train_data,
                               data.train_labels,
                               data.val_data,
                               data.val_labels,
                               init_factor=1e-3,
                               loss="mse",
                               lrn_rate=5e-1,
                               lr_decay=1e-6,
                               epochs=20
                               )
single_layer_perceptrons.plot(graph_save_path, epoch_steps=20)
single_layer_perceptrons.save(data_save_path)