
# coding: utf-8

### Face Recognition
# 
# Goal: build a face recognition system. Many of the ideas presented here are from:
# [FaceNet](https://arxiv.org/pdf/1503.03832.pdf).
# and [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf). 
# 
# Face recognition problems fall into two main categories: 
# 
# - Face Verification - "is this the claimed person?". 
# - Face Recognition - "who is this person?".
# 
# FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, can then determine if two pictures are of the same person.

#

# Load the required packages
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.set_printoptions(threshold=np.nan)


### 0) Naive Face Verification
# 
# In Face Verification, given two images  to determine if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel.
# If the distance between the raw images are less than a chosen threshold, it may be the same person
#
# * This algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, etc. 
# * Rather than using the raw image, can learn an encoding, $f(img)$.  
# * By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.

### 1) Encoding face images into a 128-dimensional vector 
# 
### 1.1) Using a ConvNet to compute encodings
# The network architecture follows the Inception model from [Szegedy *et al.*](https://arxiv.org/abs/1409.4842). 

# The key things to know:
# 
# - This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of $m$ face images) as a tensor of shape $(m, n_C, n_H, n_W) = (m, 3, 96, 96)$ 
# - It outputs a matrix of shape $(m, 128)$ that encodes each input face image into a 128-dimensional vector
# 

# Run the cell below to create the model for face images 
FRmodel = faceRecoModel(input_shape=(3, 96, 96))


# Load weights that have already been trained
print("Total Params:", FRmodel.count_params())


# ** Expected Output **
# <table>
# <center>
# Total Params: 3743280
# </center>
# </table>
# 

# By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128.
# Then use the encodings to compare two face images.
# 
# By computing the distance between two encodings and thresholding, can determine if the two pictures represent the same person.
# 
### 1.2) The Triplet Loss
#
# Training will use triplets of images (A, P, N):  
# 
# - A is an "Anchor" image--a picture of a person. 
# - P is a "Positive" image--a picture of the same person as the Anchor image.
# - N is a "Negative" image--a picture of a different person than the Anchor image.
# 
# These triplets are picked from training dataset written as $(A^{(i)}, P^{(i)}, N^{(i)})$ to denote the $i$-th training example. 
# 
# Make sure that an image A^{(i)} of an individual is closer to the Positive P^{(i)} than to the Negative image N^{(i)}) by at least a margin $\alpha$
#  
# * Useful functions: `tf.reduce_sum()`, `tf.square()`, `tf.subtract()`, `tf.add()`, `tf.maximum()`.

#

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    # Implement triplet loss to finetune and train a CNN (that's pretrained on ImageNet dataset
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha) # alpha is the margin and a hyperparameter picked manually
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

#

with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **loss**
#         </td>
#         <td>
#            528.143
#         </td>
#     </tr>
# 
# </table>

### 2) Load the pre-trained model
# 
# FaceNet is trained by minimizing the triplet loss. 

# Load the model from previously trained model
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# Now, use this model to perform face verification and face recognition! 

### 3) Applying the model
# 
# Let's build a database containing one encoding vector for each person who is allowed to enter the office.
# Use `img_to_encoding(image_path, model)`, which runs the forward propagation of the model on the specified image. 

# Run code to build the database, which maps each person's name to a 128-dimensional encoding on their face 
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)



# As presented above, use the L2 distance [np.linalg.norm](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html). 
# Note: This implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7. 
# 

#

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    # Implement the verify() function which checks if the front-door camera picture (`image_path`) is actually the person called "identity."
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. 
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image 
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open 
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

        
    return dist, door_open

#

# Run a verification algorithm on this picture

verify("images/camera_0.jpg", "younes", database, FRmodel)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **It's younes, welcome in!**
#         </td>
#         <td>
#            (0.65939283, True)
#         </td>
#     </tr>
# 
# </table>

# Run another verification on a different image

verify("images/camera_2.jpg", "kian", database, FRmodel)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **It's not kian, please go away**
#         </td>
#         <td>
#            (0.86224014, False)
#         </td>
#     </tr>
# 
# </table>

### 3.2) Face Recognition
# The face verification system is mostly working well. But since Kian got his ID card stolen, he came back to the office the next day and couldn't get in! 
# 
# To solve this, change face verification system to a face recognition system. This way, no one has to carry an ID card anymore.
# An authorized person can just walk up to the building, and the door will unlock for them! 

#

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    # Implement a face recognition system that takes as input an image, and figures out if it is one of the authorized persons (and if so, who).
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ##
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    # Initialize "min_dist" to a large value, say 100 
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. 
        if dist < min_dist:
            min_dist = dist
            identity = name

    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

#

# See if the who_it_is() algorithm identifies the person

who_is_it("images/camera_0.jpg", database, FRmodel)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **it's younes, the distance is 0.659393**
#         </td>
#         <td>
#            (0.65939283, 'younes')
#         </td>
#     </tr>
# 
# </table>

# Ways to improve the facial recognition model algorithm:
# 
# - Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database.
#       - Then given a new image, compare the new face to multiple pictures of the person. This would increase accuracy.
# - Crop the images to just contain the face, and less "border" regions around the face. This preprocessing removes some irrelevant pixels around the face, and also makes the algorithm more robust.
# 

# Summary:
# - Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem. 
# - The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
# - The same encoding can be used for verification and recognition. Measuring distances between two images' encodings helps determine whether pictures are the same person. 

# 

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model used is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - The implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
