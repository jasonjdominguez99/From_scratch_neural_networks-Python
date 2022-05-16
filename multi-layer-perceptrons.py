# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:07:44 2020

@author: Jason
"""

# 3. Multilayer Perceptrons

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
                     pad=False,
                     img_first_format=True,
                     one_hot_encode=True
                     )

# Define and use a multi layer perceptrons (MLP) model for learning the MNIST
# data, then plot the training and validation accuracy as a function of epoch
# and save this data to a csv file
input_size = data.train_data.shape[0]
hidden_size = 20
output_size = data.train_labels.shape[1]
save_dir = "C:/Users/Jason/Documents/"
save_name = "save_name"
graph_save_path = save_dir + save_name + "-graph.png"
data_save_path = save_dir + save_name + "-data.csv"

multi_layer_perceptrons = Model(input_size=input_size,
                                output_size=output_size,
                                hidden_size=hidden_size,
                                output_activation="sigmoid"
                                )
multi_layer_perceptrons.train(data.train_data,
                              data.train_labels,
                              data.val_data,
                              data.val_labels,
                              init_factor=1e-3,
                              loss="mse",
                              lrn_rate=1e-2,
                              optimizer="sgd",
                              epochs=40
                              )
multi_layer_perceptrons.plot(graph_save_path, epoch_steps=1)
multi_layer_perceptrons.save(data_save_path)