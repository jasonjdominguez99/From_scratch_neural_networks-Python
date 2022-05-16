# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:36:15 2020

@author: Jason
"""

import numpy as np


class ImageData:
    """
    Objects of the ImageData class are used for preprocessing image data and
    their corresponding labels for use in Single Perceptron, Single Layer 
    Perceptron, Multilayer Perceptron and LeNet5 task for Machine Learning
    Laboratory (Tasks 1, 2, 3, 4, respectively).
    """

    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.val_data = []
        self.val_labels = []
        self.unsorted_data1 = []
        self.unsorted_data_labels1 = []
        self.unsorted_data2 = []
        self.unsorted_data_labels2 = []
    
    
    def get_data(self, dataset=None, images1=None, image_labels1=None, 
                 images2=None, image_labels2=None):
        """
        Method that assigns the input dataset, or unsorted data and labels, to
        their corresponding object attributes.

        Parameters
        ----------
        dataset : Object of some dataset class, optional
            Dataset including sorted training and validation sets of images
            and their corresponding labels (e.g. MNIST dataset, mnist).
            The default is None.
        images1 : np.array, optional
            Array of images belonging to a single label (e.g. all images of
            zeros). The default is None.
        image_labels1 : np.array, optional
            Array of labels corresponding to images1 - all of the same label 
            (e.g. a (number_of_examples, 1) array of all zeros). The default 
            is None.
        images2 : np.array, optional
            Array of images belonging to a single label. The default is None.
        image_labels2 : np.array, optional
            Array of labels corresponding to images1 - all of the same label.
            The default is None.

        Returns
        -------
        None.

        """
        if dataset == None:
            self.unsorted_data1 = images1
            self.unsorted_data_labels1 = image_labels1
            self.unsorted_data2 = images2
            self.unsorted_data_labels2 = image_labels2
        else:
            self.train_data = dataset.train_images()
            self.train_labels = dataset.train_labels()
            self.val_data = dataset.test_images()
            self.val_labels = dataset.test_labels()
        
        
    @staticmethod
    def one_hot_encode(labels):
        """
        Static method which one-hot (or 1-to-c) encodes the input labels.
        If an example has the label is 3 and there are 5 possible label values,
        (0,1,2,3,4), the encoded label for that example will be [0,0,0,1,0]

        Parameters
        ----------
        labels : np.array
            Labels which will be one-hot encoded.

        Returns
        -------
        labels_enc : np.array
            The one-hot encoded labels.

        """
        m = labels.shape[0] # Number of example labels to be encoded
        values_to_encode = np.unique(labels)
        size = len(values_to_encode) # Number of values to be encoded
    
        labels_enc = np.zeros((m, size))
        for i in range(m):
            labels_enc[i, int(labels[i])] = 1
    
        return labels_enc
    
    
    def train_val_split(self, pre_shuffle=False):
        """
        Method that combines the unordered_data1 and unordered_data2 and 
        random splits into training and validation sets with a split ratio of 
        2/3 training, 1/3 validation. The labels are correspondingly split.

        Parameters
        ----------
        pre_shuffle : boolean, optional
            Indicates whether shuffling before splitting data is required.
            The default is False.
        Returns
        -------
        None.

        """
        # If unsorted image data is not in a flattened format it must be 
        # flatten before combining
        if len(self.unsorted_data1.shape) > 2:
            # Get height and width of images
            h1 = self.unsorted_data1.shape[1]
            w1 = self.unsorted_data1.shape[2]
            self.unsorted_data1 = self.unsorted_data1.reshape((-1, h1*w1)) 
            # Now in format (examples, height*width)
        if len(self.unsorted_data2.shape) > 2:
            h2 = self.unsorted_data2.shape[1]
            w2 = self.unsorted_data2.shape[2]
            self.unsorted_data2 = self.unsorted_data2.reshape((-1, h2*w2))   
            
        m1 = self.unsorted_data1.shape[0] 
        # Number of examples belonging to dataset 1
        m2 = self.unsorted_data2.shape[0]    
        
        # Combine images and labels for consistent shuffling, so labels stay
        # with their associated image
        data_with_labels1 = np.hstack((self.unsorted_data_labels1,
                                       self.unsorted_data1))
        data_with_labels2 = np.hstack((self.unsorted_data_labels2,
                                       self.unsorted_data2))
        
        if pre_shuffle:
            # Want the particular images in training and validation sets to be
            # random each time
            np.random.shuffle(data_with_labels1)
            np.random.shuffle(data_with_labels2)
        
        # 2/3 of total images and their labels will be put into training set,
        # the rest into validation set
        train_ratio = 2/3
        train_split1 = int(train_ratio * m1)
        train_split2 = int(train_ratio * m2)
        
        train_labels1 = data_with_labels1[:train_split1,
                                          0].reshape(train_split1, 1)
        train_data1 = data_with_labels1[:train_split1, 1:]
        val_labels1 = data_with_labels1[train_split1:,
                                        0].reshape(m1 - train_split1, 1)
        val_data1 = data_with_labels1[train_split1:, 1:]
        
        train_labels2 = data_with_labels2[:train_split2,
                                          0].reshape(train_split2, 1)
        train_data2 = data_with_labels2[:train_split2, 1:]
        val_labels2 = data_with_labels2[train_split2:,
                                        0].reshape(m2 - train_split2, 1)
        val_data2 = data_with_labels2[train_split2:, 1:]
        
        self.train_data = np.vstack((train_data1, train_data2))
        self.train_labels = np.vstack((train_labels1, train_labels2))
        self.val_data = np.vstack((val_data1, val_data2))
        self.val_labels = np.vstack((val_labels1, val_labels2))
        
        # Return images to their original unflatten format (they'll be
        # flattened again in image_preprocess method if required)
        self.train_data = self.train_data.reshape((self.train_data.shape[0],
                                                   h1, w1))
        self.val_data = self.val_data.reshape((self.val_data.shape[0], 
                                               h2, w2))
    
    
    def image_preprocess(self, split_data=False, pre_shuffle=False,
                         normalize=True, flatten=False, pad=False,
                         img_first_format=True):        
        """
        Method which preprocesses the train_data and val_data attributes of 
        the ImageData object. Possible preprocessing includes: splitting 
        unordered data into training and validation sets, flattening images, 
        padding images and reshaping into (image, examples) array format.

        Parameters
        ----------
        split_data : boolean, optional
            Indicates whether splitting the data into training and validation
            sets is required. The default is False.
        pre_shuffle : boolean, optional
            Indicates whether shuffling before splitting data is required.
            The default is False.
        normalize : boolean, optional
            Indicates whether normalizing the images by 1/255 is required.
            The default is True.
        flatten : boolean, optional
            Indicates whether flattening the images from (height, width) into
            a (height*width) vector is required. The default is False.
        pad : boolean, optional
            Indicates whether padding the images is required.
            The default is False.
        img_first_format : boolean, optional
            Indicates whether image first format (image, examples) is required. 
            The alternative is image last format (examples, image).
            The default is True.

        Returns
        -------
        None.

        """
        # If data is not yet sorted, split and sort into training and 
        # validation sets
        if split_data:
            self.train_val_split(pre_shuffle)
        
        # Normalize images
        if normalize:
            self.train_data = self.train_data/255
            self.val_data = self.val_data/255
        
        # Flatten images (no further flattening or dimension expansion needed
        # if data was split into training and validation sets)
        if flatten:
            img_height = self.train_data.shape[1]
            img_width = self.train_data.shape[2]
            self.train_data = self.train_data.reshape((-1,
                                                       img_height*img_width))
            self.val_data = self.val_data.reshape((-1, img_height*img_width))
        elif not flatten:
            # If not flattening images, need a 4D-array of size
            # (examples, img_height, imag_width, n_channels) for Keras input
            self.train_data = np.expand_dims(self.train_data, 3)
            self.val_data = np.expand_dims(self.val_data, 3)
        
        # Pad images with a 2 pixel thick border of zeros
        if pad:
            self.train_data = np.pad(self.train_data,((0,0),(2,2),(2,2),(0,0)))
            self.val_data = np.pad(self.val_data,((0,0),(2,2),(2,2),(0,0)))
            
        # Get images into the correct format, either (examples, images) or
        # (images, examples)
        if img_first_format:
            self.train_data = self.train_data.T
            self.val_data = self.val_data.T    
        
        print("\nPreprocessed images shapes:")
        print("Training images shape: " + str(self.train_data.shape))
        print("Validation images shape: " + str(self.val_data.shape))
    
    
    def label_preprocess(self, one_hot_encode=True):
        """
        Method which preprocesses the train_labels and val_labels attributes
        of the ImageData object. This may include one-hot-encoding the labels
        if required.

        Parameters
        ----------
        one_hot_encode : boolean, optional
            Indicates whether labels should be one-hot-encoded. 
            The default is True.

        Returns
        -------
        None.

        """
        # Ensure that labels are in the required (examples, label) format
        m_train = len(self.train_labels)
        m_val = len(self.val_labels)
        self.train_labels = self.train_labels.reshape((m_train, 1))
        self.val_labels = self.val_labels.reshape((m_val, 1))
        
        # One-hot encode the labels
        if one_hot_encode:
            # Resultant labels array will be of size (examples, number of 
            # unique labels)
            self.train_labels = self.one_hot_encode(self.train_labels)
            self.val_labels = self.one_hot_encode(self.val_labels)
        
        print("\nPreprocessed labels shapes:")
        print("Training labels shape: " + str(self.train_labels.shape))
        print("Validation labels shape: " + str(self.val_labels.shape))
            
            
    def data_preprocess(self, split_data=False, pre_shuffle=False,
                        normalize=False, flatten=False, pad=False,
                        img_first_format=True, one_hot_encode=False):
        """
        Method that performs image and label preprocessing in sequence to get
        data and labels consistent and ready for any machine learning using
        this data.

        Parameters
        ----------
        split_data : boolean, optional
            Indicates whether splitting the data into training and validation
            sets is required. The default is False.
        pre_shuffle : boolean, optional
            Indicates whether shuffling before splitting data is required.
            The default is False.
        normalize : boolean, optional
            Indicates whether normalizing the images by 1/255 is required.
            The default is True.
        flatten : boolean, optional
            Indicates whether flattening the images from (height, width) into
            a (height*width) vector is required. The default is False.
        pad : boolean, optional
            Indicates whether padding the images is required. 
            The default is False.
        img_first_format : boolean, optional
            Indicates whether image first format (image, examples) is required. 
            The alternative is image last format (examples, image).
            The default is True.
        one_hot_encode : boolean, optional
            Indicates whether labels should be one-hot-encoded. 
            The default is True.

        Returns
        -------
        None.

        """
        self.image_preprocess(split_data, pre_shuffle, normalize,
                              flatten, pad, img_first_format)
        self.label_preprocess(one_hot_encode)

