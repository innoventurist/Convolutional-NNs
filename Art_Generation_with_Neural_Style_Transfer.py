
# coding: utf-8

### Deep Learning & Art: Neural Style Transfer
# 
# Goal: learn about Neural Style Transfer. This algorithm was created by [Gatys et al. (2015).] --> (https://arxiv.org/abs/1508.06576)

# 

### Import Packages

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
get_ipython().magic('matplotlib inline')


### 1) Problem Statement ###
# 
# Neural Style Transfer (NST) is one of the most fun techniques in deep learning. It merges two images, namely:
# - a **"content" image (C) and a "style" image (S), to create a "generated" image (G). 
#-  The generated image G combines the "content" of the image C with the "style" of image S. 

### 2) Transfer Learning
# 
# Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that.
# Transfer learning: The idea of using a network trained on a different task and applying it to a new task. 
# 
# Reference:
# [original NST paper] --> (https://arxiv.org/abs/1508.06576), for using the VGG network. Specifically, using VGG-19, a 19-layer version of the VGG network.
# This model has already been trained on the very large ImageNet database, and has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers). 

# Run the following code to load parameters from the VGG model,specifically VGG-19.
pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)

# ```

### 3) Neural Style Transfer (NST) ###
# 
# Will build the Neural Style Transfer (NST) algorithm in three steps:
# 
# - Build the content cost function $J_{content}(C,G)$
# - Build the style cost function $J_{style}(S,G)$
# - Put it together to get $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$. 
# 
### 3.1) Computing the content cost
#

# Run code to see a picture of the Louvre
content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);

#

### 3.1.1) Make generated image G match the content of image C ###
#
# Compute the "content cost" using TensorFlow. 
# 
# The 3 steps to implement this function are:
# 1. Retrieve dimensions from `a_G`: 
#     - To retrieve dimensions from a tensor `X`, use: `X.get_shape().as_list()`
# 2. Unroll `a_C` and `a_G` using these functions:
#           - [tf.transpose] --> (https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/transpose)
#           - and [tf.reshape] --> (https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/reshape).
# 3. To Compute the content cost, use these functions:
#           - [tf.reduce_sum] --> (https://www.tensorflow.org/api_docs/python/tf/reduce_sum),
#           - [tf.square] --> (https://www.tensorflow.org/api_docs/python/tf/square)
#           - [tf.subtract] --> (https://www.tensorflow.org/api_docs/python/tf/subtract).

#

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar to compute using equation 1 above.
    """
    # "Generate" image G to have similar content as the input image C
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list() # height, width and channels of the chosen hidden layers
    
    # Reshape a_C and a_G 
    a_C_unrolled = tf.reshape(a_C, shape = [m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape = [m, n_H * n_W, n_C])
    
    # compute the cost with tensorflow 
    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))/(4 * n_H * n_W * n_C)
    
    return J_content


#

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J_content**
#         </td>
#         <td>
#            6.76559
#         </td>
#     </tr>
# 
# </table>

#
### 3.2) Computing the style cost

# Use the following style image
style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image);


# This was painted in the style of [impressionism] --> (https://en.wikipedia.org/wiki/Impressionism).
#

### 3.2.1) Style matrix ###
#
# * Other functions: [matmul](https://www.tensorflow.org/api_docs/python/tf/matmul) and [transpose](https://www.tensorflow.org/api_docs/python/tf/transpose).

#

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    # Using TensorFlow, implement a function that computes the Gram matrix of a matrix A 
    GA = tf.matmul(A, tf.transpose(A)) # Multiplying two matrices
    
    return GA


# 

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = \n" + str(GA.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **GA**
#         </td>
#         <td>
#            [[  6.42230511  -4.42912197  -2.09668207] <br>
#  [ -4.42912197  19.46583748  19.56387138] <br>
#  [ -2.09668207  19.56387138  20.6864624 ]]
#         </td>
#     </tr>
# 
# </table>

### 3.2.2) Style cost
# 
# The 3 steps to implement this function are:
# 1. Retrieve dimensions from the hidden layer activations a_G: 
#     - To retrieve dimensions from a tensor X, use: `X.get_shape().as_list()`
# 2. Unroll the hidden layer activations a_S and a_G into 2D matrices.
#     - Can use [tf.transpose](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/transpose) and [tf.reshape](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/reshape).
# 3. Compute the Style matrix of the images S and G. (Use the function previously written.) 
# 4. Compute the Style cost:
# - [tf.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/reduce_sum), [tf.square](https://www.tensorflow.org/api_docs/python/tf/square) and [tf.subtract](https://www.tensorflow.org/api_docs/python/tf/subtract) useful.

#

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    # Compute the style cost for a single layer
    # Minimize the distance between the Gram matrix of the "style" image S and the gram matrix of the "generated" image G.
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(GS - GG)) / (4 * n_C**2 * (n_W * n_H)**2) 
    
    return J_style_layer


# 

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    print("J_style_layer = " + str(J_style_layer.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J_style_layer**
#         </td>
#         <td>
#            9.19028
#         </td>
#     </tr>
# 
# </table>

### 3.2.3) Style Weights ###
# 
# * So far have captured the style from only one layer. 
# * Will get better results if "merge" style costs from several different layers. 
# * Each layer will be given weights ($\lambda^{[l]}$) that reflect how much each layer will contribute to the style.
# * By default, will give each layer equal weight, and the weights add up to 1.  ($\sum_{l}^L\lambda^{[l]} = 1$)

# 

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]



### Exercise: compute style cost ###
# 
# * Have implemented a compute_style_cost(...) function. 
# * It calls  `compute_layer_style_cost(...)` several times, and weights their results using the values in `STYLE_LAYERS`.

# 

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


### 3.3) Defining the total cost to optimize

# Finally, let's create a cost function that minimizes both the style and the content cost. 

# 

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    # Total cost function to minimize both style and content cost 
    J = alpha * J_content + beta * J_style 
    
    return J


#

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J**
#         </td>
#         <td>
#            35.34667875478276
#         </td>
#     </tr>
# 
# </table>

#
## 4) Solving the optimization problem

# Finally, let's put everything together to implement Neural Style Transfer!

### 1. Start the interactive session.
# "[Interactive Session] --> (https://www.tensorflow.org/api_docs/python/tf/InteractiveSession)". 
# 

# Reset the graph ###
tf.reset_default_graph()

# Start interactive session ###
sess = tf.InteractiveSession()


### 2. Load the content image
content_image = scipy.misc.imread("images/louvre_small.jpg") # Load the "content" image (the Louvre museum picture)
content_image = reshape_and_normalize_image(content_image)   # reshape and normalize the "content" image 


### 3. Load the style image 
style_image = scipy.misc.imread("images/monet.jpg")          # Load the "style" image (Claude Monet's painting)
style_image = reshape_and_normalize_image(style_image)       # Reshape and normalize the "style" image


### 4. Generated image correlated with content image
# This will help the content of the "generated" image more rapidly match the content of the "content" image. 
generated_image = generate_noise_image(content_image) # Initialize the "generated" image as a noisy image created from the content_image
imshow(generated_image[0]);


### 5. Load pre-trained VGG19 model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat") # Load the VGG19 model.


### Content Cost ###
# To get the program to compute the content cost, now assign `a_C` and `a_G` to be the appropriate hidden layer activations. 

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)


### Style cost ###

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


### Total cost ###

# Compute the total cost J by calling `total_cost()`. 
J = total_cost(J_content, J_style, alpha = 10, beta = 40)


### Optimizer ###
# * Use the Adam optimizer to minimize the total cost `J`.
# * Use a learning rate of 2.0.  
# * [Adam Optimizer documentation] --> (https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

# 

# define optimizer
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step 
train_step = optimizer.minimize(J)


### Implement the model ###
### Reference for [assign] --> (https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/assign) 

#

def model_nn(sess, input_image, num_iterations = 200): #function initializes the variables of the tensorflow graph
    
    # Initialize global variables (need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(input_image)) # is the input of the VGG19 model
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step) # used for a large number of steps
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

#

# Run cell to generate an artistic image. Takes about 3min on CPU for every 20 iterations but can start observing attractive results after ≈140 iterations.
model_nn(sess, generated_image)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **Iteration 0 : **
#         </td>
#         <td>
#            total cost = 5.05035e+09 <br>
#            content cost = 7877.67 <br>
#            style cost = 1.26257e+08
#         </td>
#     </tr>
# 
# </table>

# Finished! 
# 
# Image seen should be like this --> <img src="images/louvre_generated.png" style="width:800px;height:300px;">

### 6) Conclusion
## Summary:
# - Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
# - It uses representations (hidden layer activations) based on a pretrained ConvNet. 
# - The content cost function is computed using one hidden layer's activations.
# - The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
# - Optimizing the total cost function results in synthesizing new images. 
#

# ### References:
# 
# The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" was also an inspiration.
# The pre-trained network used in this implementation is a VGG network is due to Simonyan and Zisserman (2015). Pre-trained weights were from the MathConvNet team. 
# 
# - Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) 
# - Harish Narayanan, [Convolutional neural networks for artistic style transfer.](https://harishnarayanan.org/writing/artistic-style-transfer/)
# - Log0, [TensorFlow Implementation of "A Neural Algorithm of Artistic Style".](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
# - Karen Simonyan and Andrew Zisserman (2015). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
# - [MatConvNet.](http://www.vlfeat.org/matconvnet/pretrained/)
# 

# In[ ]:



