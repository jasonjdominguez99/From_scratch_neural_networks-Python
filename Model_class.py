# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:15:09 2020

@author: Jason
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Model:   
    """
    Objects of the Model class can be used to train a single perceptron,
    single layer of perceptrons or two-level multilayer perceptron machine 
    learning model/network.
    """
    
    def __init__(self, input_size, output_size, hidden_size=None,
                 output_activation="sigmoid"):
        """
        Initialization method called upon creating an object of the class.

        Parameters
        ----------
        input_size : int
            The number of input features for the network.
        output_size : int
            The number of output values from the network. e.g for a binary 
            classification this will be 1 (0 or 1), but for n class 
            classification it will be n.
        hidden_size : int, optional
            If a MLP network is required, this is the number of perceptron in
            the hidden layer. The default is None.
        output_activation : string, optional
            The activation function for the output perceptrons. For binary 
            classification, "sign" can be used. For single layer perceptrons
            or MLP, use "sigmoid". The default is "sigmoid".

        Returns
        -------
        None.

        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.train_acc_log = []
        self.val_acc_log = []
        
    
    def initialize_parameters(self, factor):
        """
        Method to initialize the weights and biases of the model network.
        Initialization is random, from a normal distribution between factor
        and -factor. If the model is initialized with no hidden layer, then
        only one weight matrix and bias vector is intialized. If not, two
        weight matrices and bias vectors are initialized corresponding to 
        the hidden and output layers.

        Parameters
        ----------
        factor : float
            If using random intialization this number scales the random
            value to be between -factor and factor.

        Returns
        -------
        W : np.array
            Weight matrix corresponding to output layer, of size (n_in, n_out).
        b : np.array
            Bias vector corresponding to the output layer, of size (1, n_out).
        Or 
        W1 : np.array
            Weight matrix corresponding to hidden layer,
            of size (n_in, n_hidden).
        b1 : np.array
            Bias vector corresponding to the hidden layer,
            of size (1, n_hidden).
        W2 : np.array
            Weight matrix corresponding to output layer,
            of size (n_hidden, n_out).
        b2 : np.array
            Bias vector corresponding to the output layer, of size (1, n_out).

        """
        n_in = self.input_size # Number of input features
        n_hidden = self.hidden_size # Number of hidden layer perceptrons
        n_out = self.output_size # Number of output perceptrons
        
        if n_hidden == None:
            # Network only has an output layer
            W = np.random.randn(n_in, n_out)*factor
            b = np.random.randn(1, n_out)*factor

            return W, b
           
        else:
            # Network has a hidden layer and output layer 
            W1 = np.random.randn(n_in, n_hidden)*factor
            b1 = np.random.randn(1, n_hidden)*factor
            W2 = np.random.randn(n_hidden, n_out)*factor
            b2 = np.random.randn(1, n_out)*factor

            return W1, b1, W2, b2
        
    
    @staticmethod
    def sigmoid(x):
        """
        Method that computes the sigmoid of the input x.

        Parameters
        ----------
        x : np.array
            Input array for which the element-wise sigmoid is required.

        Returns
        -------
        s : np.array
            Array corresponding to the element-wise sigmoid of the elements
            of x.

        """
        s = 1/(1 + np.exp(-x))
        return s
    
    
    def forward_propagation(self, data, W1, b1, W2=None, b2=None):
        """
        Computes the forward propagation through the model using one or more
        training examples and the weights and biases of the model.

        Parameters
        ----------
        data : np.array
            DESCRIPTION.
        W1 : np.array
            Weight matrix corresponding to first layer.
        b1 : np.array
            Bias vector corresponding to the first layer.
        W2 : np.array, optional
            Weight matrix corresponding to second layer, if MLP.
            The default is None.
        b2 : np.array, optional
            Bias vector corresponding to the second layer, if MLP. 
            The default is None.

        Returns
        -------
        y1 : np.array or float
            Output of first layer of model for the image(s) in data.
        y2 : np.array
            Output of second layer of model for the image(s) in data., if MLP.

        """
        m = data.shape[1] # Number of examples in the input data
        hidden = self.hidden_size # Number of hidden perceptrons in the 
                                  # network
        outputs = self.output_size # Number of output perceptrons in the
                                   # network
        
        if hidden == None:
            # Network only has an output layer
            # Apply the chosen activation function to the linear function of
            # the form x_transpose W + b, where x is the input data
            if self.output_activation == "sign":
                y1 = np.sign(np.dot(data.T, W1) + b1).reshape(m, outputs) 
                # Reshapes are used to ensure after manipulation that the 
                # array shape is correct
            elif self.output_activation == "sigmoid":
                y1 = self.sigmoid(np.dot(data.T, W1) + b1).reshape(m, outputs)
            
            return y1
        else:
            # Network has a hidden layer and output layer 
            y1 = self.sigmoid(np.dot(data.T, W1) + b1).reshape(m, hidden)
                
            # Outputs of the output perceptrons are computed as the sigmoid of
            # the linear function hidden_activation W2 + b2
            y2 = self.sigmoid(np.dot(y1, W2) + b2).reshape(m, outputs)
            
            return y1, y2
            
    
    def update_weights(self, y1, x, d, loss, lrn_rate,
                       W1, b1, W2=None, b2=None, y2=None):
        """
        Updates the the current weights and biases of the model using the
        stochastic or batch gradient descent learning rule. This can be done
        for log loss or MSE loss.

        Parameters
        ----------
        y1 : np.array or float
            y1 : np.array or float
            Output of first layer of model for the image(s) in x.
        x : np.array
            Image or images used for updating the weights.
        d : np.array or int
            True labels of image/images in x.
        loss : string
            The loss function used for defining the weight update rules. Can 
            be either "log_loss" or mean-square error loss, "mse.
        lrn_rate : float
            The learning rate for the weight update rule.
            Indicates how large steps are taken.
        W1 : np.array
            Current weights for the first layer of the model.
        b1 : np.array
            Current bias(es) for the first layer of the model.
        W2 : np.array, optional
            Current weights for the second layer of the model, if MLP.
            The default is None.
        b2 : np.array, optional
            Current biases for the second layer of the model, if MLP.
            The default is None.
        y2 : np.array, optional
            Output of second layer of model for the image(s) in x, if MLP
            The default is None.

        Returns
        -------
        W_new : np.array
            Updated weights for the first layer of the model.
        b_new : np.array
            Updated biases for the first layer of the model.
        or
        W1_new : np.array
            Updated weights for the first layer of the model, if MLP.
        b1_new : np.array
            Updated biases for the first layer of the model, if MLP.
        W2_new : np.array
            Updated weights for the second layer of the model, if MLP.
        b2_new : np.array
            Updated biases for the second layer of the model, if MLP.

        """
        m = x.shape[1]
        
        if self.hidden_size == None:
            # Network only has an output layer
            # Single layer perceptron
            diff = d - y1
            
            # Perform the weight update rule depending on what
            # loss function is being used
            if loss == "log_loss":
                W_new = W1 + lrn_rate*(1/m)*np.dot(x, diff)
                b_new = b1 + lrn_rate*(1/m)*np.sum(diff,
                                                   axis=0, keepdims=True)
            elif loss == "mse":
                W_new = W1 + lrn_rate*(1/m)*np.dot(x, diff*y1*(1 - y1))
                b_new = b1 + lrn_rate*(1/m)*np.sum(diff*y1*(1 - y1),
                                                   axis=0, keepdims=True)
            
            return W_new, b_new
        else:
            # Network has a hidden layer and output layer (MLP)
            diff = d - y2
            
            # Perform the weight update rule depending on what
            # loss function is being used
            if loss == "log_loss":
                W2_new = W2 + lrn_rate*np.dot(y1.T, diff).reshape(W2.shape)
                b2_new = b2 + lrn_rate*np.sum(
                    diff, axis=0, keepdims=True).reshape(b2.shape)
                W1_new = W1 + lrn_rate*np.dot(
                    x, np.dot(diff, W2.T)*y1*(1 - y1)).reshape(W1.shape)
                b1_new = b1 + lrn_rate*np.sum(
                    np.dot(diff, W2.T)*y1*(1 - y1),
                    axis=0, keepdims=True).reshape(b1.shape)
            elif loss == "mse":
                W2_new = W2 + lrn_rate*np.dot(
                    y1.T, diff*y2*(1 - y2)
                    ).reshape(W2.shape)
                b2_new = b2 + lrn_rate*np.sum(
                    diff*y2*(1 - y2), axis=0, keepdims=True).reshape(b2.shape)
                W1_new = W1 + lrn_rate*np.dot(
                    x,np.dot(diff*y2*(1 - y2), W2.T)*y1*(1 - y1)
                    ).reshape(W1.shape)
                b1_new = b1 + lrn_rate*np.sum(
                    np.dot(diff*y2*(1 - y2), W2.T)*y1*(1 - y1),
                    axis=0, keepdims=True).reshape(b1.shape)
            
            return W1_new, b1_new, W2_new, b2_new
    
    
    def evaluate(self, epoch,  train_or_val, data, true_labels,
                 W1, b1, W2=None, b2=None):
        """
        Evaluate the performance (accuracy) of the model on the entire
        training or validation dataset, using the current weights and biases 
        of the model. Values of the accuracy are stored in arrays for writing 
        to a csv file after training and plotting graphs.

        Parameters
        ----------
        epoch : int
            At which epoch the accuracy is being evaluated, used for storing
            accuracies for plotting later.
        train_or_val : string
            Indicated whether that data is training or validation, used for
            storing accuracies for plotting later. Possible values are 
            "train" or "val".
        data : np.array
            Image data, either entire training or validation set.
        true_labels : np.array
            The true labels corresponding to images in data.
        W1 : np.array
            Current weights for the first layer of the model.
        b1 : np.array
            Current bias(es) for the first layer of the model.
        W2 : np.array, optional
            Current weights for the second layer of the model, if MLP.
            The default is None.
        b2 : np.array, optional
            Current biases for the second layer of the model, if MLP.
            The default is None.

        Returns
        -------
        acc : float
            Accuracy of model predictions on all data images, between 0 and 1.

        """
        m = data.shape[1]
        
        if self.hidden_size == None:
            # Network only has an output layer, no hidden layer
            # Single layer perceptron
            output = self.forward_propagation(data, W1, b1)  
        else:
            # Network has a hidden layer and output layer (MLP)
            _, output = self.forward_propagation(data, W1, b1, W2, b2)
            
        if output.shape[1] > 1:
                # For multiclass problems, one-hot-encoded labels are used, so
                # to compute accuracy argmax must see which element is maximum
                # and hence the correct label.
                # e.g. output = [0,0,1,0] ----> output_labels = [2]
                output = np.argmax(output, axis = 1).reshape(m, 1)
                true_labels = np.argmax(true_labels, axis = 1).reshape(m, 1)    
        
        # Compute the accuracy
        acc = float(np.sum(output == true_labels))/m
            
        # Record and display the accuracy
        if train_or_val == "train":
            # Forward propagation using training data
            # Store the training accuracy in train_acc_log for plotting etc
            # after training
            self.train_acc_log.insert(epoch - 1, acc)
                    
        elif train_or_val == "val":
            # Forward propagation using validation data
            # Store the validation accuracy in val_acc_log for plotting etc
            # after training
            self.val_acc_log.insert(epoch - 1, acc)
            # Display the prediction vectors and true label vectors
            #print("True labels:")
            #print(true_labels)
            #print("Output:")
            #print(output)
            
        return acc
    
    
    def train(self, train_data, train_labels, val_data, val_labels,
              init_factor=1e-3, loss = "log_loss", lrn_rate=0.01, lr_decay=None,
              optimizer="sgd", epochs=20, max_accept_error=0, print_epochs=1):
        """
        Train the model for either a set number of epochs or until a maximum
        acceptable error has been reached. Training is done via forward
        and backward propagation through the model and using gradient-based
        weight update.

        Parameters
        ----------
        train_data : np.array
            The images used for training the network, of shape
            (features, examples).
        train_labels : np.array
            Labels corresponding to the images in train_data, of
            shape (examples, n_output).
        val_data : np.array
            The images used for validation of the network, of shape
            (features, examples).
        val_labels : np.array
            Labels corresponding to the images in val_data, of
            shape (examples, n_output).
        init_factor : float, optional
            For weight intialization this number scales the random values to
            be between 0 and factor.
            The default is 1e-3.
        loss : string, optional
            The loss function used for defining the weight update rules. Can 
            be either "log_loss" or mean-square error loss, "mse. 
            The default is "log_loss".
        lrn_rate : float, optional
            The learning rate for the weight update rule.
            Indicates how large steps are taken. The default is 0.01.
        lr_decay : float, optional
            The value used in the weight decay equation to gradually reduce
            the learning rate from its initial value after each weight update.
            The default is None.
        optimizer : string, optional
            The optimization technique used for updating the weights. Can
            be either batch gradient descent "batch_gd", looking at all
            training images before updating the weights, or stochastic
            gradient descent "sgd", looking at a single training images, then
            updating the weights. The default is "sgd".
        epochs : int, optional
            Number of iterations through whole dataset should
            before training is complete. The default is 20.
        max_accept_error : float, optional
            When training error reaches this value, training will stop.
            The default is 0.
        print_epochs : int, optional
            Number of how often the accuracies and epoch number
            should be displayed. The default is 1.

        Returns
        -------
        None.

        """
        m = train_data.shape[1]
        initial_lrate = lrn_rate # This is needed if using learing rate decay 
                                 # so as to not ovewrite the initial learning
                                 # rate
        
        # Initialize the weights of the network
        if self.hidden_size == None:
            # Network only has an output layer
            W, b = self.initialize_parameters(init_factor)
        else:
            # Network has a hidden layer and output layer
            W1, b1, W2, b2 = self.initialize_parameters(init_factor)
        
        
        print("\nTraining has started...")
        i = 0 # Variable to increment upon weight updates, used for weight 
              # decay equation
        # Each epoch use all the training data to update the weights of the
        # network
        for epoch in range(1, epochs + 1):
            
            if epoch%print_epochs == 0:
                # Only print the interation number every print_epochs
                # iteration
                print("Epoch " + str(epoch) + ":")
            
            if self.hidden_size == None:
                # Network has no hidden layer, just an output
                if optimizer == "sgd":
                    for example in np.random.permutation(m):
                        # Select images in random order
                        # Forward propagation to compute output
                        # Feed in an image at a time and update the weights
                        train_example = train_data[:, example].reshape(
                            train_data.shape[0], 1)
                        train_label = train_labels[example, :].reshape(
                            1,train_labels.shape[1])
                        y = self.forward_propagation(train_example,
                                                     W, b)
                        # Backward propagation to update weights
                        if lr_decay != None:
                            # Using learning rate decay, so calculated reduced
                            # learning rate based on what iteration it is
                            lrn_rate = initial_lrate*(1/(1 + lr_decay*i))
                        W, b = self.update_weights(y, train_example,
                                                   train_label, 
                                                   loss, lrn_rate, W, b)
                        i += 1
                elif optimizer == "batch_gd":
                    # Update weights using all examples at a time
                    # Forward propagation to compute output
                    y = self.forward_propagation(train_data, W, b)
                    # Backward propagation to update weights
                    W, b = self.update_weights(y, train_data, train_labels,
                                               loss, lrn_rate, W, b)  
                  
                # Calculate, store and display the training and
                # validation accuracy each epoch   
                train_acc = self.evaluate(epoch, "train", train_data,
                                          train_labels, W, b)
                val_acc = self.evaluate(epoch, "val", val_data,
                                        val_labels, W, b)
                
                if epoch%print_epochs == 0:
                    # Only display the accuracy every print_epochs interation
                    print("train_accuracy = " + str(train_acc))
                    print("validation_accuracy = " + str(val_acc))
                train_error = 1 - train_acc
                if train_error <= max_accept_error:
                    # When max_accept_error is given exit the epoch loop 
                    # prematurely based on whether the error is small enough
                    break
                    
            else:
                # Network has a hidden layer
                if optimizer == "sgd":
                    for example in np.random.permutation(m):
                        # Select images in random order
                        # Forward propagation to compute output
                        # Feed in an image at a time and update the weights
                        train_example = train_data[:,example].reshape(
                            train_data.shape[0], 1)
                        train_label = train_labels[example, :].reshape(
                            1, train_labels.shape[1])
                        y1, y2 = self.forward_propagation(train_example,
                                                          W1, b1, W2, b2)
                        # Backward propagation to update weights
                        if lr_decay != None:
                            # Using learning rate decay, so calculated reduced
                            # learning rate based on what iteration it is
                            lrn_rate = initial_lrate*(1/(1 + lr_decay*i))
                        W1, b1, W2, b2 = self.update_weights(y1, train_example,
                                                             train_label,
                                                             loss, lrn_rate, 
                                                             W1, b1,
                                                             W2, b2, y2)
                        i += 1
                elif optimizer == "batch_gd":
                    # Update weights using all examples at a time
                    # Forward propagation to compute output
                    y1, y2 = self.forward_propagation(train_data, 
                                                      W1, b1, W2, b2)
                    # Backward propagation to update weights
                    W1, b1, W2, b2 = self.update_weights(y1, train_data, 
                                                         train_labels,
                                                         loss, lrn_rate, 
                                                         W1, b1,
                                                         W2, b2, y2)
                    
                # Calculate, store and display the training and validation
                # accuracy each epoch and return error to compare with
                # max_accept_error, if given    
                train_acc = self.evaluate(epoch, "train", train_data,
                                          train_labels, W1, b1, W2, b2)
                val_acc = self.evaluate(epoch, "val", val_data,
                                        val_labels, W1, b1, W2, b2)
                
                if epoch%print_epochs == 0:
                    # Only display the accuracy every print_epochs interation
                    print("train_accuracy = " + str(train_acc))
                    print("validation_accuracy = " + str(val_acc))
                train_error = 1 - train_acc
                if train_error <= max_accept_error:
                    # When max_accept_error is given exit the epoch loop
                    # prematurely based on whether the error is small enough
                    break
        
        print("\nTraining has finished")
        print("\nFinal training accuracy: " + str(self.train_acc_log[-1]))
        print("\nFinal validation accuracy: " + str(self.val_acc_log[-1]))
        
    
    def plot(self, file_path, epoch_steps=1):
        """
        Method to plot the training and validation accuracy, 
        for each epoch, against the epoch number and save the image.

        Parameters
        ----------
        file_path : string
            File path where the csv will be saved.
        epoch_steps : int, optional
            The interval steps on the x axis of the graph.
            The default is 1.

        Returns
        -------
        None.

        """
        # Retrieve the training accuracy and validation accuracy data
        acc = self.train_acc_log
        val_acc = self.val_acc_log

        epochs = range(1, len(acc) + 1)
        
        # Plot the graph
        plt.figure(figsize=(20,15))
        plt.plot(epochs, acc, 'r', label="Training accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
        plt.legend(loc=0, fontsize=18)
        plt.grid(True)
        plt.xticks(np.arange(0, epochs[-1] + 1, step=epoch_steps), 
                   fontsize=18)
        plt.xlim(1, epochs[-1] + 1)
        plt.xlabel("Epochs", fontsize=20)
        plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=18)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy", fontsize=20)
        
        # Save the file and display
        plt.savefig(file_path)
        print("\nTraining and validation accuracy graph printed successfully!")
        plt.show()
        
    
    def save(self, file_path):
        """
        Method to save training and validation accuracy data,
        for each epoch, as a csv file.

        Parameters
        ----------
        file_path : string
            File path where the csv will be saved.

        Returns
        -------
        None.

        """
        # Retrieve the training accuracy and validation accuracy
        acc = np.array(self.train_acc_log).reshape(len(self.train_acc_log), 1)
        val_acc = np.array(self.val_acc_log).reshape(len(self.val_acc_log), 1)
        training_log = np.hstack((acc, val_acc))
        
        # Store data into a pandas dataframe and save to a csv file
        column_headers = ["Training accuracy", "Validation accuracy"]
        df = pd.DataFrame(data = training_log, columns = column_headers)
        df.index += 1
        df.to_csv(file_path)
        print("\nData saved in a csv file to file path successfully!")