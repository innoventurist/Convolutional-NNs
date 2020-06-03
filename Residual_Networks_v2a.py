
# coding: utf-8

# # Residual Networks
# 
# Learn how to build very deep convolutional networks, using Residual Networks (ResNets).
# Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allows to train much deeper networks
# In theory, very deep networks can represent very complex functions; but in practice, they are hard to train.
# 
# Goal:
# - Implement the basic building blocks of ResNets. 
# - Put together these building blocks to implement and train a state-of-the-art neural network for image classification. 

# 

# Repository will be done in Keras. 
# 
# Run to load the required packages.
import numpy as np
from keras import layers            # Helps minimize the number of user actions for common use cases and clear understanding of error messages
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
get_ipython().magic('matplotlib inline')

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


### 1) The problem of very deep neural networks
# 
# Neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers.
# Vanishing gradient: The speed of learning decreases very rapidly for the shallower layers as the network trains.
# 
# Solve this problem by building a Residual Network!

### 2) Building a Residual Network
# 
# In ResNets, a "shortcut" or a "skip connection" allows the model to skip layers:  
# 
# By stacking ResNet blocks on top of each other, can form a very deep network. 
# Having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function.
# This means it can stack on additional blocks with little risk of harming training set performance.  
#
# Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different.
# Will implement both of them: the "identity block" and the "convolutional block."
# 
# - Skip connection "skips over" 2 layers. 
# - The upper path is the "shortcut path." The lower path is the "main path."
# - Have also made explicit the CONV2D and ReLU steps in each layer. To speed up training, have also added a BatchNorm step.
#

# Implement a slightly more powerful version of this identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers. 
def identity_block(X, f, filters, stage, block): # Used in ResNet, corresponding to where input activation have the same dimensions as output activation
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch' # res = residual
    bn_name_base = 'bn' + str(stage) + block + '_branch' # bn = bottleneck ; a fast array function
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. Will need later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path : first Conv2D
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) # normalizing the 'channels' axis
    X = Activation('relu')(X) # use of eELU activation
    
    # Second component of main path : second Conv2D
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X) # normalizing the 'channels' axis
    X = Activation('relu')(X) # use of ReLU activation

    # Third component of main path : third Conv2D
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X) # normalizing the 'channels' axis

    # Final step: Add shortcut value to main path, and pass it through a ReLU activation 
    X = Add()([X, X_shortcut]) # `X_shortcut` and the output from the 3rd layer `X` are added together
    X = Activation('relu')(X)  # apply ReLU activation function with no name or hyperparameters
    
    
    return X


# 

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **out**
#         </td>
#         <td>
#            [ 0.94822985  0.          1.16101444  2.747859    0.          1.36677003]
#         </td>
#     </tr>
# 
# </table>

### 2.2) The convolutional block
# 
# The ResNet "convolutional block" is the second block type. Used when the input and output dimensions don't match up.
# 
# The CONV2D layer on the shortcut path does not use any non-linear activation function.
#              - Main role is to just a (learned) linear function that reduces the dimension of the input so the dimensions match up for the later addition step. 
#

#

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s, s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X) # used to resice input x to a different dimension
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) # normalizing the 'channels' axis
    X = Activation('relu')(X) # # Apply the ReLU activation function (has no name and no hyperparameters) 

    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X) # normalizing the 'channels' axis
    X = Activation('relu')(X) # Apply the ReLU activation function (has no name and no hyperparameters) 

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X) # normalizing the 'channels' axis

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut) # normalizing the 'channels' axis

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut]) # shortcut and mainpath values added
    X = Activation('relu')(X)  # apply ReLU activation function with no name or hyperparameters
    
    
    return X


# 

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **out**
#         </td>
#         <td>
#            [ 0.09018463  1.23489773  0.46822017  0.0367176   0.          0.65516603]
#         </td>
#     </tr>
# 
# </table>

### 3) - Building ResNet model (50 layers)
# 
# Have the necessary blocks to build a very deep ResNet. "ID BLOCK" stands for "Identity block," and "ID BLOCK x3" should stack 3 identity blocks together.
# - The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
# - The 'flatten' layer doesn't have any hyperparameters or name.
# - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be `'fc' + str(classes)`.
# 
# Use this function: 
# - Average pooling [see reference](https://keras.io/layers/pooling/#averagepooling2d)
# 
# Other functions used in the code below:
# - Conv2D: [See reference](https://keras.io/layers/convolutional/#conv2d)
# - BatchNorm: [See reference](https://keras.io/layers/normalization/#batchnormalization) (axis: Integer, the axis that should be normalized (typically the features axis))
# - Zero padding: [See reference](https://keras.io/layers/convolutional/#zeropadding2d)
# - Max pooling: [See reference](https://keras.io/layers/pooling/#maxpooling2d)
# - Fully connected layer: [See reference](https://keras.io/layers/core/#dense)
# - Addition: [See reference](https://keras.io/layers/merge/#add)

#

### ResNet50 ###

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    # Implement ResNet50 model
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input) # pads the input with a pad of (3,3)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X) # applied to the 'channels' axis of the input
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X) # uses a (3,3) window and a (2,2) stride

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'b')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'c')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'd')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'b')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'c')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'd')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'e')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'b')
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'c')

    # AVGPOOL. Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size = (2, 2), name = 'avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model. 

# 
# Run the following code to build the model's graph
model = ResNet50(input_shape = (64, 64, 3), classes = 6)

#If implementation is not correct, will know by checking accuracy when running `model.fit(...)` below.

# Configure the learning process by compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#

# The model is now ready to be trained. 

# Load the SIGNS Dataset.
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#

# Run the following cell to train the model on 2 epochs with a batch size of 32. On a CPU, it should take around 5min per epoch. 
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             ** Epoch 1/2**
#         </td>
#         <td>
#            loss: between 1 and 5, acc: between 0.2 and 0.5, although your results can be different from ours.
#         </td>
#     </tr>
#     <tr>
#         <td>
#             ** Epoch 2/2**
#         </td>
#         <td>
#            loss: between 1 and 5, acc: between 0.2 and 0.5, you should see your loss decreasing and the accuracy increasing.
#         </td>
#     </tr>
# 
# </table>

# 

# Now, see how this model (trained on only two epochs) performs on the test set.
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **Test Accuracy**
#         </td>
#         <td>
#            between 0.16 and 0.25
#         </td>
#     </tr>
# 
# </table>

# Have trained the model for just two epochs, so it will achieves poor performances.

# Can optionally train the ResNet for more iterations and get a lot better performance, but this will take more than an hour when training on a CPU (GPU no more than 1min)

# 

# Can load and run our trained model on the test set
model = load_model('ResNet50.h5') 

# Prediction of the model
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ResNet50 is a powerful model for image classification when it is trained for an adequate number of iterations.
# Can apply it classification problems to perform state-of-the-art accuracy.
#

### 4) Test Another image (Optional)

# 

img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
print('Input image shape:', x.shape)
my_image = scipy.misc.imread(img_path)
imshow(my_image)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))

# 

# Print a summary of the model
model.summary()

# 

# Run the code below to visualize ResNet50.

plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))


# Summary:
# - Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.  
# - The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function. 
# - There are two main types of blocks: The identity block and the convolutional block. 
# - Very deep Residual Networks are built by stacking these blocks together.

### References: 
# 
# This repository presents the ResNet algorithm due to He et al. (2015). The implementation here also took significant inspiration and follows the structure given in the GitHub repository of Francois Chollet: 
# - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
# - Francois Chollet's GitHub repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# 

# 



