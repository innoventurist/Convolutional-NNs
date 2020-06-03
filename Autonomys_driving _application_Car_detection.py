
# coding: utf-8

### Autonomous driving: Car detection
# 
# Goal: learn about object detection using the very powerful YOLO model. Many ideas are described in the two YOLO papers:
# [Redmon et al., 2016](https://arxiv.org/abs/1506.02640) and [Redmon and Farhadi, 2016](https://arxiv.org/abs/1612.08242).
#
# Car detection dataset:
# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Drive.ai Sample Dataset</span> (provided by drive.ai) is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. We are grateful to Brody Huval, Chih Hu and Rahul Patel for  providing this data. 
# 
# Learn To:
# - Use object detection on a car detection dataset
# - Deal with bounding boxes

### Import libraries ###

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

get_ipython().magic('matplotlib inline')


# Note: Keras's backend has been imported as as K. This means that to use a Keras function, will need to write: `K.function(...)`.

### 1) Problem Statement
#
# Learn how "You Only Look Once" (YOLO) performs object detection, and then apply it to car detection.
# The YOLO model is computationally expensive to train, so load pre-trained weights to use. 

### 2) YOLO ("You Only Look Once")

# YOLO is a popular algorithm because it achieves high accuracy while also running in real-time.
# This algorithm's sense "only looks once" at the image is the idea of only one forward propagation pass through the network to make predictions.
# After non-max suppression, it then outputs recognized objects together with the bounding boxes.
# 
### Anchor Boxes ###
# * Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.


##Useful references:
#     * [Keras argmax] --> (https://keras.io/backend/#argmax)
#     * [Keras max] -->(https://keras.io/backend/#max)
#     * [boolean mask] --> (https://www.tensorflow.org/api_docs/python/tf/boolean_mask)  

#

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6): 
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    # Apply a filter by thresholding to get rid of any box for which the class "score" is less than a chosen threshold
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes to keep (with probability >= threshold)
    filtering_mask = ((box_class_scores) >= threshold)
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask, name = 'boolean_mask')
    boxes = tf.boolean_mask(boxes, filtering_mask, name = 'boolean_mask')
    classes = tf.boolean_mask(box_classes, filtering_mask, name = 'boolean_mask')
    
    return scores, boxes, classes


# In[3]:

with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **scores[2]**
#         </td>
#         <td>
#            10.7506
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes[2]**
#         </td>
#         <td>
#            [ 8.42653275  3.27136683 -0.5313437  -4.94137383]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes[2]**
#         </td>
#         <td>
#            7
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **scores.shape**
#         </td>
#         <td>
#            (?,)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes.shape**
#         </td>
#         <td>
#            (?, 4)
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes.shape**
#         </td>
#         <td>
#            (?,)
#         </td>
#     </tr>
# 
# </table>

### 2.3) Non-max suppression ###
# Non-max suppression uses the very important function called "Intersection over Union", or IoU.

#

### IoU ###

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """
    # Implement IoU. The (0,0) origin starts at the top left corner of the image. As x increases, move to the right.  As y increases, move down.
    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0],box2[0]) # maximum of the x1 coordinates of the two boxes
    yi1 = max(box1[1],box2[1]) # maximum of the y1 coordinates of the two boxes
    xi2 = min(box1[2],box2[2]) # minimum of the x2 coordinates of the two boxes
    yi2 = min(box1[3],box2[3]) # minimum of the y2 coordinates of the two boxes
    inter_width = yi2 - yi1
    inter_height = xi2 - xi1
    inter_area = max(inter_width, 0)* max(inter_height, 0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou


# 

## Test case 1: boxes intersect
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4) 
print("iou for intersecting boxes = " + str(iou(box1, box2)))

## Test case 2: boxes do not intersect
box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))

## Test case 3: boxes intersect at vertices only
box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))

## Test case 4: boxes intersect at edge only
box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))


# **Expected Output**:
# 
# ```
# iou for intersecting boxes = 0.14285714285714285
# iou for non-intersecting boxes = 0.0
# iou for boxes that only touch at vertices = 0.0
# iou for boxes that only touch at edges = 0.0
# ```

### YOLO non-max suppression
# 
# Now ready to implement non-max suppression. The key steps are: 
# 1. Select the box that has the highest score.
# 2. Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= `iou_threshold`).
# 3. Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.
# 
# References: 
# 
# - [tf.image.non_max_suppression()] --> (https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
# - [K.gather()] --> (https://www.tensorflow.org/api_docs/python/tf/keras/backend/gather)  

#

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes wanted
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    # Implement yolo_non_max_suppression() using TensorFlow.
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes kept
    nms_indices = tf.image.non_max_suppression(boxes = boxes, scores = scores, max_output_size = max_boxes, iou_threshold = iou_threshold)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes


# 

with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **scores[2]**
#         </td>
#         <td>
#            6.9384
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes[2]**
#         </td>
#         <td>
#            [-5.299932    3.13798141  4.45036697  0.95942086]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes[2]**
#         </td>
#         <td>
#            -2.24527
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **scores.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes.shape**
#         </td>
#         <td>
#            (10, 4)
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
# 
# </table>

### 2.4 Wrapping up the filtering

#

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) for predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this code uses (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes wanted
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    # Implement a function taking the output of the deep CNN and filter through all the boxes using the functions just implemented.
    # Retrieve outputs of the YOLO model 
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[0], yolo_outputs[1], yolo_outputs[2], yolo_outputs[3]

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions previously implemented to perform Score-filtering with a threshold of score_threshold 
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold 
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    
    
    return scores, boxes, classes


# 

with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **scores[2]**
#         </td>
#         <td>
#            138.791
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes[2]**
#         </td>
#         <td>
#            [ 1292.32971191  -278.52166748  3876.98925781  -835.56494141]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes[2]**
#         </td>
#         <td>
#            54
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **scores.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes.shape**
#         </td>
#         <td>
#            (10, 4)
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
# 
# </table>

### Summary for YOLO:
# - Input image (608, 608, 3)
# - The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
# - After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
#     - Each cell in a 19x19 grid over the input image gives 425 numbers. 
#     - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes. 
#     - 85 = 5 + 80 where 5 is because (p_c, b_x, b_y, b_h, b_w) has 5 numbers, and 80 is the number of classes want to detect
# - Can select only few boxes based on:
#     - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
#     - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
# - This gives YOLO's final output. 

## 3) Test YOLO pre-trained model on images ###

# In this part, use a pre-trained model and test it on the car detection dataset. 

# Session to execute the computation graph and evaluate the tensors.
sess = K.get_session()


### 3.1) Defining classes, anchors and image shape ###
# 

# Read class names and achors from text files. The dataset has 720x1280 images which was already pre-processed into 608x608 images. 
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    


### 3.2) Loading a pre-trained model ###
# * Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. 

# Run the cell below to load the model from this file.
yolo_model = load_model("model_data/yolo.h5") # Loads the weights of a trained YOLO model

#

# Summary of the layers the model contains.

yolo_model.summary()

#

## 3.3) Convert output of the model to usable bounding box tensors

# `yolo_head` function definition in the file ['keras_yolo.py'] --> (https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py).

# Run cell to have tensor pass through non-trivial processing and conversion
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names)) # have added yolo_outputs to graph and gives all predicted boxes of yolo_model in correct format

# This set of 4 tensors is ready to be used as input by `yolo_eval` function.

### 3.4) Filtering boxes ###

# Use previously implemented `yolo_eval`. 

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape) # to perform filtering and select only the best boxes


### 3.5) Run the graph on an image ###
# Will need to run a TensorFlow session, to have it compute `scores, boxes, classes`.

# 

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    sess --  tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """
    # Implement predict() which runs the graph to test YOLO on an image.
    # Preprocess image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # Need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes

#

# Run this cell on the "test.jpg" image to verify that the function is correct.
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **Found 7 boxes for test.jpg**
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.60 (925, 285) (1045, 374)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.66 (706, 279) (786, 350)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **bus**
#         </td>
#         <td>
#            0.67 (5, 266) (220, 407)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.70 (947, 324) (1280, 705)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.74 (159, 303) (346, 440)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.80 (761, 282) (942, 412)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.89 (367, 300) (745, 648)
#         </td>
#     </tr>
# </table>

# 
### Summary:
#     
# - YOLO is a state-of-the-art object detection model that is fast and accurate
# - It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume. 
# - The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
# - Would filter through all the boxes using non-max suppression. Specifically: 
#     - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
#     - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
# - Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset and lot of computation, have used previously trained model parameters in this repository.
#     - Can also try fine-tuning the YOLO model with own dataset, though this would be a fairly non-trivial exercise. 

# References: The ideas here came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository.

# Predictions of the YOLO model on pictures taken from a camera while driving around the Silicon Valley in courtest of:
# [drive.ai](https://www.drive.ai/) for providing this dataset

# The pre-trained weights used in this exercise came from the official YOLO website. 
# - Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
# - Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
# - Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
# - The official YOLO website (https://pjreddie.com/darknet/yolo/) 

# 



