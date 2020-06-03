
# coding: utf-8

### Convolutional Neural Networks
# 
# Goal: Implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation. 
# 
# Notes:
# - Superscript [l] denotes an object of the l^{th}$layer. 
#
# - Superscript $(i)$ denotes an object from the $i^{th}$ example. 
# 
# - Subscript $i$ denotes the $i^{th}$ entry of a vector.    
#     
# - n_H, n_W, and n_C denote respectively the height, width and number of channels of a given layer. If referencing a specific layer l, can also write n_H^{[l]}, n_W^{[l]}, n_C^{[l]}. 
# - n_{H_{prev}}, n_{W_{prev}}, and n_{C_{prev}} denote the height, width and number of channels of the previous layer.
# If referencing a specific layer l, this could also be denoted n_H^{[l-1]}, n_W^{[l-1]}, n_C^{[l-1]}. 
# 
#
### 1) Packages

import numpy as np                  # fundamental package for scientific computing in Python
import h5py 
import matplotlib.pyplot as plt     # Library to plot graphs in Python

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)                   # Used to keep all the random function cells consistent 


### 2) Outline of the Assignment
# 
# Implement the building blocks of a convolutional neural network! Each function will have detailed instructions for the steps needed:
#     
# Will implement these functions from scratch in `numpy`.
# 
# Note: for every forward function, there is its corresponding backward equivalent. Hence, at every step of forward module, will store some parameters in a cache.
# These parameters are used to compute gradients during backpropagation. 

### 3) Convolutional Neural Networks
# 
# Although programming frameworks make convolutions easy to use, they remain one of the hardest concepts to understand in Deep Learning.
# A convolution layer transforms an input volume into an output volume of different size, as shown below. 
# 
# 
# Now, will build every step of the convolution layer. First implement two helper functions:
#  - one for zero padding and the other for computing the convolution function itself. 

### 3.1) Zero-Padding
# 
# Zero-padding adds zeros around the border of an image.
# 
# The main benefits of padding are:
# 
# - Can use a CONV layer without necessarily shrinking the height and width of the volumes.
# - Important for building deeper networks, since otherwise the height/width would shrink going to deeper layers.
# - It helps keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.
#
# [Use np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html).
#

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    # Implement function that pads all images of a batch of examples X with zeros
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0)) 
    

    return X_pad


#

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1,1])
print ("x_pad[1,1] =\n", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])


# **Expected Output**:
# 
# ```
# x.shape =
#  (4, 3, 3, 2)
# x_pad.shape =
#  (4, 7, 7, 2)
# x[1,1] =
#  [[ 0.90085595 -0.68372786]
#  [-0.12289023 -0.93576943]
#  [-0.26788808  0.53035547]]
# x_pad[1,1] =
#  [[ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]
#  [ 0.  0.]]
# ```

### 3.2) Single step of convolution 
# 
# Here, implement a single step of convolution, . This will be used to build a convolutional unit, which: 
# 
# - Takes an input volume 
# - Applies a filter at every position of the input
# - Outputs another volume (usually of different size)
#
# Later, this function will be applied to multiple positions of the input to implement the full convolutional operation. 
# 
# Implement conv_single_step(). [Hint](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html).
#

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Implement a single-step convolution, applying the filter to a single position of the input.
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    
    return Z

# 

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)


# **Expected Output**:
# <table>
#     <tr>
#         <td>
#             **Z**
#         </td>
#         <td>
#             -6.99908945068
#         </td>
#     </tr>
# 
# </table>

### 3.3) Convolutional Neural Networks - Forward pass
# 
# In the forward pass, will take many filters and convolve them on the input. Each 'convolution' gives a 2D matrix output. Then, stack these outputs to get a 3D volume. 
# 
# Finally can also have access to the hyperparameters dictionary which contains the stride and the padding. 
#
# Remember:
# The formulas relating the output shape of the convolution to the input shape is:
# $$ n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
# $$ n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
# $$ n_C = \text{number of filters used in the convolution}$$
# 

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    #  Implement the function below to convolve the filters `W` on an input activation `A_prev`.
    # Retrieve dimensions from A_prev's shape, the activation output from the previous layer   
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape 
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Compute the dimensions of the CONV output volume using the formula given above. Use int() to apply the 'floor' operation. 
    n_H = int((n_H_prev - f + 2*pad)/stride + 1)
    n_W = int((n_W_prev - f + 2*pad)/stride + 1)
    
    # Initialize the output volume Z with zeros. 
    Z = np.zeros([m, n_H, n_W, n_C])
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                 # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]     # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" 
            vert_start = h * stride
            vert_end = h * stride + f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" 
                horiz_start = w * stride
                horiz_end = w * stride + f
                
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). 
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. 
                    weights = W[:,:,:, c]
                    biases = b[:,:,:, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                                        
    
    # Making sure output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


# In[7]:

np.random.seed(1)
A_prev = np.random.randn(10,5,7,4)
W = np.random.randn(3,3,4,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =\n", np.mean(Z))
print("Z[3,2,1] =\n", Z[3,2,1])
print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])


# **Expected Output**:
# ```
# Z's mean =
#  0.692360880758
# Z[3,2,1] =
#  [ -1.28912231   2.27650251   6.61941931   0.95527176   8.25132576
#    2.31329639  13.00689405   2.34576051]
# cache_conv[0][1][2][3] = [-1.1191154   1.9560789  -0.3264995  -1.34267579]
# ``` 

### 4) Pooling layer 
# 
# The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, and make feature detectors more invariant to its position in the input.
# The two types of pooling layers are: 
# 
# - Max-pooling layer: slides an ($f, f$) window over the input and stores the max value of the window in the output.
# - Average-pooling layer: slides an ($f, f$) window over the input and stores the average value of the window in the output.
# 
# Note:
# As there's no padding, the formulas binding the output shape of the pooling to the input shape is:
# 
# $$ n_H = \lfloor \frac{n_{H_{prev}} - f}{stride} \rfloor +1 $$
# 
# $$ n_W = \lfloor \frac{n_{W_{prev}} - f}{stride} \rfloor +1 $$
# 
# $$ n_C = n_{C_{prev}}$$

#

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    # Implement forward pass of the pooling layer.
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    

    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h * stride
            vert_end = h * stride + f
            
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                horiz_start = w * stride
                horiz_end = w * stride + f
                
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, :, c]                    
                    # Compute the pooling operation on the slice. 
                    # Use an if statement to differentiate the modes. 
                    # Use np.max and np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache


# 

# Case 1: stride of 1
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A =\n", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A =\n", A)


# ** Expected Output**
# ```
# mode = max
# A.shape = (2, 3, 3, 3)
# A =
#  [[[[ 1.74481176  0.90159072  1.65980218]
#    [ 1.74481176  1.46210794  1.65980218]
#    [ 1.74481176  1.6924546   1.65980218]]
# 
#   [[ 1.14472371  0.90159072  2.10025514]
#    [ 1.14472371  0.90159072  1.65980218]
#    [ 1.14472371  1.6924546   1.65980218]]
# 
#   [[ 1.13162939  1.51981682  2.18557541]
#    [ 1.13162939  1.51981682  2.18557541]
#    [ 1.13162939  1.6924546   2.18557541]]]
# 
# 
#  [[[ 1.19891788  0.84616065  0.82797464]
#    [ 0.69803203  0.84616065  1.2245077 ]
#    [ 0.69803203  1.12141771  1.2245077 ]]
# 
#   [[ 1.96710175  0.84616065  1.27375593]
#    [ 1.96710175  0.84616065  1.23616403]
#    [ 1.62765075  1.12141771  1.2245077 ]]
# 
#   [[ 1.96710175  0.86888616  1.27375593]
#    [ 1.96710175  0.86888616  1.23616403]
#    [ 1.62765075  1.12141771  0.79280687]]]]
# 
# mode = average
# A.shape = (2, 3, 3, 3)
# A =
#  [[[[ -3.01046719e-02  -3.24021315e-03  -3.36298859e-01]
#    [  1.43310483e-01   1.93146751e-01  -4.44905196e-01]
#    [  1.28934436e-01   2.22428468e-01   1.25067597e-01]]
# 
#   [[ -3.81801899e-01   1.59993515e-02   1.70562706e-01]
#    [  4.73707165e-02   2.59244658e-02   9.20338402e-02]
#    [  3.97048605e-02   1.57189094e-01   3.45302489e-01]]
# 
#   [[ -3.82680519e-01   2.32579951e-01   6.25997903e-01]
#    [ -2.47157416e-01  -3.48524998e-04   3.50539717e-01]
#    [ -9.52551510e-02   2.68511000e-01   4.66056368e-01]]]
# 
# 
#  [[[ -1.73134159e-01   3.23771981e-01  -3.43175716e-01]
#    [  3.80634669e-02   7.26706274e-02  -2.30268958e-01]
#    [  2.03009393e-02   1.41414785e-01  -1.23158476e-02]]
# 
#   [[  4.44976963e-01  -2.61694592e-03  -3.10403073e-01]
#    [  5.08114737e-01  -2.34937338e-01  -2.39611830e-01]
#    [  1.18726772e-01   1.72552294e-01  -2.21121966e-01]]
# 
#   [[  4.29449255e-01   8.44699612e-02  -2.72909051e-01]
#    [  6.76351685e-01  -1.20138225e-01  -2.44076712e-01]
#    [  1.50774518e-01   2.89111751e-01   1.23238536e-03]]]]
# ```

#

# Case 2: stride of 2
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A =\n", A)
print()

A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A =\n", A)


# **Expected Output:**
#     
# ```
# mode = max
# A.shape = (2, 2, 2, 3)
# A =
#  [[[[ 1.74481176  0.90159072  1.65980218]
#    [ 1.74481176  1.6924546   1.65980218]]
# 
#   [[ 1.13162939  1.51981682  2.18557541]
#    [ 1.13162939  1.6924546   2.18557541]]]
# 
# 
#  [[[ 1.19891788  0.84616065  0.82797464]
#    [ 0.69803203  1.12141771  1.2245077 ]]
# 
#   [[ 1.96710175  0.86888616  1.27375593]
#    [ 1.62765075  1.12141771  0.79280687]]]]
# 
# mode = average
# A.shape = (2, 2, 2, 3)
# A =
#  [[[[-0.03010467 -0.00324021 -0.33629886]
#    [ 0.12893444  0.22242847  0.1250676 ]]
# 
#   [[-0.38268052  0.23257995  0.6259979 ]
#    [-0.09525515  0.268511    0.46605637]]]
# 
# 
#  [[[-0.17313416  0.32377198 -0.34317572]
#    [ 0.02030094  0.14141479 -0.01231585]]
# 
#   [[ 0.42944926  0.08446996 -0.27290905]
#    [ 0.15077452  0.28911175  0.00123239]]]]
# ```

# Have now implemented the forward passes of all the layers of a convolutional network. 
# 

# ## 5 - Backpropagation in convolutional neural networks
# 
# In modern deep learning frameworks, only have to implement the forward pass, and the framework takes care of the backward pass, so most deep learning engineers don't need to bother with details of backward pass.
#
# In convolutional neural networks, can calculate the derivatives with respect to the cost in order to update the parameters.  
#

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Implement backward pass for conv layer function to sum over all the training examples, filters, heights, and widths.
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.random.randn( *A_prev.shape)                          
    dW = np.random.randn( *W.shape)
    db = np.random.randn( *b.shape)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                         # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h 
                    vert_end = vert_start + f
                    horiz_start = w  
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, : ]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpadded da_prev_pad
        dA_prev[i, :, :, :] = da_prev_pad[ pad:-pad, pad:-pad, : ]
    
    # Making sure output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


# 

# Run conv_forward to initialize the 'Z' and 'cache_conv",
# which will use to test the conv_backward function
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

# Test conv_backward
dA, dW, db = conv_backward(Z, cache_conv)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))


# ** Expected Output: **
# <table>
#     <tr>
#         <td>
#             **dA_mean**
#         </td>
#         <td>
#             1.45243777754
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **dW_mean**
#         </td>
#         <td>
#             1.72699145831
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **db_mean**
#         </td>
#         <td>
#             7.83923256462
#         </td>
#     </tr>
# 
# </table>
# 

### 5.2 Pooling layer) backward pass
# 

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    # Create a "mask" matrix that keeps track of where the maximum of the matrix is.
    mask = (x == np.max(x)) 
    
    return mask


#

np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)


# **Expected Output:** 
# 
# <table> 
# <tr> 
# <td>
# 
# **x =**
# </td>
# 
# <td>
# 
# [[ 1.62434536 -0.61175641 -0.52817175] <br>
#  [-1.07296862  0.86540763 -2.3015387 ]]
# 
#   </td>
# </tr>
# 
# <tr> 
# <td>
# **mask =**
# </td>
# <td>
# [[ True False False] <br>
#  [False False False]]
# </td>
# </tr>
# 
# 
# </table>

### 5.2.2) Average pooling - backward pass 
# 
# In max pooling, for each input window, all the "influence" on the output came from a single input value--the max.
# In average pooling, every element of the input window has equal influence on the output. So to implement backprop, implement a helper function that reflects this.
# 
# (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ones.html)

# Implement the function below to equally distribute a value dz through a matrix of dimension shape.
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which needed to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which distributes the value of dz
    """

    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / (n_H * n_W)
    
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape) * average
    
    return a


# 

a = distribute_value(2, (2,2))
print('distributed value =', a)


# **Expected Output**: 
# 
# <table> 
# <tr> 
# <td>
# distributed_value =
# </td>
# <td>
# [[ 0.5  0.5]
# <br\> 
# [ 0.5  0.5]]
# </td>
# </tr>
# </table>

### 5.2.3) Putting it together: Pooling backward 
# 
# Now have everything needed to compute backward propagation on a pooling layer.
# 

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode that would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    
    # Retrieve information from cache 
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" 
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape 
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros 
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                         # loop over the training examples
        
        # select training example from A_prev 
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" 
                    vert_start = h 
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev 
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice 
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) 
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        
                        # Get the value a from dA 
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf 
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. 
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    
    # Making sure output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev


# 

np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1]) 


# **Expected Output**: 
# 
# mode = max:
# <table> 
# <tr> 
# <td>
# 
# **mean of dA =**
# </td>
# 
# <td>
# 
# 0.145713902729
# 
#   </td>
# </tr>
# 
# <tr> 
# <td>
# **dA_prev[1,1] =** 
# </td>
# <td>
# [[ 0.          0.        ] <br>
#  [ 5.05844394 -1.68282702] <br>
#  [ 0.          0.        ]]
# </td>
# </tr>
# </table>
# 
# mode = average
# <table> 
# <tr> 
# <td>
# 
# **mean of dA =**
# </td>
# 
# <td>
# 
# 0.145713902729
# 
#   </td>
# </tr>
# 
# <tr> 
# <td>
# **dA_prev[1,1] =** 
# </td>
# <td>
# [[ 0.08485462  0.2787552 ] <br>
#  [ 1.26461098 -0.25749373] <br>
#  [ 1.17975636 -0.53624893]]
# </td>
# </tr>
# </table>

# Summary:
# Can now understand how convolutional neural networks work.
# Have implemented all the building blocks of a neural network. Next repository implements a ConvNet using TensorFlow.

# 



