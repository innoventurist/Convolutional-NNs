
# coding: utf-8

# # Convolutional Neural Networks with TensorFlow
# 
# Goal:
# 
# - Implement helper functions that you will use when implementing a TensorFlow model
# - Implement a fully functioning ConvNet using TensorFlow
# - Overall, should be able to build and train a ConvNet in TensorFlow for a classification problem
#
# ## 1.0 - TensorFlow model
# 
# First, load in the packages. 

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf   # Most practical applications, which has many built-in functions to simply call
from tensorflow.python.framework import ops
from cnn_utils import *

# Get Python interface to the libmagic file type identification library.
get_ipython().magic('matplotlib inline')
np.random.seed(1)

#

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.

# 

# Show example of a labelled image in the dataset.
index = 6
plt.imshow(X_train_orig[index]) # Can change the value of `index` below and re-run to see different examples
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# Since this is an image dataset, it is more natural to apply a ConvNet to it.

# Examine the shapes of the data
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


### 1.1) Create placeholders
# 
# TensorFlow requires creating placeholders for the input data that will be fed into the model when running the session.
# 
# Next, implement the function below to create placeholders for the input image X and the output Y. Should not define the number of training examples for the moment.
# Look at "tf.placeholder documentation"](https://www.tensorflow.org/api_docs/python/tf/placeholder).
#

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    # Create placeholders for X and Y with dimensions given
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    
    return X, Y


# 

X, Y = create_placeholders(64, 64, 3, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


# **Expected Output**
# 
# <table> 
# <tr>
# <td>
#     X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
# 
# </td>
# </tr>
# <tr>
# <td>
#     Y = Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)
# 
# </td>
# </tr>
# </table>

### 1.2) Initialize parameters
#
### tf.get_variable() :
# [Search for the tf.get_variable documentation](https://www.tensorflow.org/api_docs/python/tf/get_variable).  Notice that the documentation says:

#

### initialize_parameters ###

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Note that we will hard code the shape values in the function to make the grading simpler.
    Normally, functions should take values as inputs rather than hard coding.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    
    # Initialize weight parameters and filters with Xavier initializer   
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters


# In[8]:

tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1[1,1,1] = \n" + str(parameters["W1"].eval()[1,1,1]))
    print("W1.shape: " + str(parameters["W1"].shape))
    print("\n")
    print("W2[1,1,1] = \n" + str(parameters["W2"].eval()[1,1,1]))
    print("W2.shape: " + str(parameters["W2"].shape))


# ** Expected Output:**
# 
# ```
# W1[1,1,1] = 
# [ 0.00131723  0.14176141 -0.04434952  0.09197326  0.14984085 -0.03514394
#  -0.06847463  0.05245192]
# W1.shape: (4, 4, 3, 8)
# 
# 
# W2[1,1,1] = 
# [-0.08566415  0.17750949  0.11974221  0.16773748 -0.0830943  -0.08058
#  -0.00577033 -0.14643836  0.24162132 -0.05857408 -0.19055021  0.1345228
#  -0.22779644 -0.1601823  -0.16117483 -0.10286498]
# W2.shape: (2, 2, 8, 16)
# ```

### 1.3) Forward propagation

# 

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Note that for simplicity and grading purposes, we'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Implement the `forward_propagation` function below to build the following model: 'CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED`.
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    F = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)

    return Z3


# Clears the default graph stack and reset the global default graph
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = \n" + str(a))


# **Expected Output**:
# 
# ```
# Z3 = 
# [[-0.44670227 -1.57208765 -1.53049231 -2.31013036 -1.29104376  0.46852064]
#  [-0.17601591 -1.57972014 -1.4737016  -2.61672091 -1.00810647  0.5747785 ]]
# ```

### 1.4) Compute cost ###

# By adjusting the weights of the network to reduce the cost, the neural network can improve its predictions.

#

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    # Implement the compute cost function below.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)) # computes the mean of elements for both softmax activation function and resulting loss
    
    return cost

# 

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = " + str(a))


# **Expected Output**: 
# ```
# cost = 2.91034
# ```

### 1.5) Model 
# 
# The model below should:
# 
# - create placeholders
# - initialize parameters
# - forward propagate
# - compute the cost
# - create an optimizer
# 
# Finally, will create a session and run a for loop for num_epochs, get the mini-batches, and then for each mini-batch to optimize the function.
# [Document for initializing the variables] --> (https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer)
#
# Notes:
# For details, check out the documentation for [Adam Optimizer] --> (https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
# More information on sess.run in documentation [tf.Sesssion#run] --> (https://www.tensorflow.org/api_docs/python/tf/Session#run).

#

# Merge helper function above to build a model
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed) # returns a list of minibatches

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                """
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost.
                # The feedict should contain a minibatch for (X,Y).
                """
                
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters

#

# Run the following cell to train the model for 100 epochs. Check if cost after epoch 0 and 5 matches the output. If not, stop the cell and check code!
_, _, parameters = model(X_train, Y_train, X_test, Y_test)


# **Expected output**: although it may not match perfectly, the expected output should be close to expected and cost value should decrease.
# 
# <table> 
# <tr>
#     <td> 
#     **Cost after epoch 0 =**
#     </td>
# 
#     <td> 
#       1.917929
#     </td> 
# </tr>
# <tr>
#     <td> 
#     **Cost after epoch 5 =**
#     </td>
# 
#     <td> 
#       1.506757
#     </td> 
# </tr>
# <tr>
#     <td> 
#     **Train Accuracy   =**
#     </td>
# 
#     <td> 
#       0.940741
#     </td> 
# </tr> 
# 
# <tr>
#     <td> 
#     **Test Accuracy   =**
#     </td>
# 
#     <td> 
#       0.783333
#     </td> 
# </tr> 
# </table>


# Summary:
# Have now built a model that recognizes SIGN language with almost 80% accuracy on the test set.
# Can actually improve its accuracy by spending more time tuning the hyperparameters, or using regularization (as this model clearly has a high variance). 

# 



