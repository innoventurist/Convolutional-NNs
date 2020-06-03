
# coding: utf-8

### Keras tutorial
# 
# Goal:
# 1. Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK. 
# 2. See how to use it and build a deep learning algorithm.
# 
### Why Keras? 
# 
# * Keras was developed to enable deep learning engineers to build and experiment with different models very quickly. 
# * Keras is an even higher-level framework and provides additional abstractions. 
# * Being able to go from idea to result with the least possible delay is key to finding good models. 
# * However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models that are still implemented in TensorFlow instead of Keras. 
# * Overall, Keras will work fine for many common models.

#

### Load the required packages
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().magic('matplotlib inline')


# Note: Have imported a lot of functions from Keras. Can use them by calling them directly in your code.
# 
# Don't have to create the graph and then make a separate `sess.run()` call to evaluate those variables.

### 1) Emotion Tracking
# 
# To build and train this model, gather pictures of some volunteers. The dataset is labeled.
#

# Run the following code to normalize the dataset and learn about its shapes.
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# Details of the "Face" dataset:
# - Images are of shape (64,64,3)
# - Training: 600 pictures
# - Test: 150 pictures

### 2) Building a model in Keras
# 
# Keras is very good for rapid prototyping. Will be able build a model that achieves outstanding results.
#

# 

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    

    Returns:
    model -- a Model() instance in Keras"""
    
 
    # Define the input placeholder as a tensor with shape input_shape. Think of this as the input image!
    X_input = Input(input_shape) # height, width and channels as a tuple

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)  # constructor call which creates an object

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates the Keras model instance, will use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    
    return model


# Have built a function to describe your model. To train and test this model, there are four steps in Keras:
  
### Step 1: create the model.    

# Create a model by calling the function above
happyModel = HappyModel(X_train.shape[1:4]


### Step 2: compile the model
 
# Optimizers to try and include `'adam'`, `'sgd'` or others.  Documentation for [optimizers](https://keras.io/optimizers/)  
# Documentation for [losses](https://keras.io/losses/)

# Compile the model by calling 'model.compile'
happyModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


### Step 3: train the model

# Traub the model on training data
happyModel.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 50)


### Step 4: evaluate model   

# Test the model on test data; Use `'X_test'` and `'Y_test'` variables to evaluate the model's performance.
preds = happyModel.evaluate(x = X_test, y = Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


## Expected performance:   
# If `happyModel()` function worked, its accuracy should be better than random guessing (50% accuracy).
# 
### Tips for improving your model:
# 
# To achieve a very good accuracy (>= 80%):
#
# until height and width dimensions are quite low and the number of channels quite large (≈32 for example).  
# Can then flatten the volume and use a fully-connected layer.
# - Use MAXPOOL after such blocks.  It will help lower the dimension in height and width.
# - Change optimizer. 'adam' works well. 
# - If having memory issues, lower your batch_size (e.g. 12 )
# - Run more epochs until the training accuracy no longer improves. 
# 

### 3) Conclusion
# - Keras is a tool recommended for rapid prototyping. allowing to quickly try out different model architectures.
# - Remember The four steps in Keras: 
#  
# 1. Create  
# 2. Compile  
# 3. Fit/Train  
# 4. Evaluate/Test  

### 4 - Test another image (Optional) 
                        
img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))


### 5) Other useful functions in Keras
# 
# Two other useful basic features of Keras: `model.summary()` and `plot_model()`           
# 
# Run the following codes.

# Print the details of layers in a table with the sizes of its inputs and outputs
happyModel.summary()


# Plot graph in a nice layout
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg')) # Save it to possibily share it on social media platforms
# It will be saved in "File" then "Open..." in the upper bar of the notebook.

# 



