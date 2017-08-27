#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_2017_08_24_23_16_32_332.jpg "Center Lane Driving"
[image3]: ./examples/center_2017_08_25_02_09_42_618.jpg "Recovery Image"
[image4]: ./examples/center_2017_08_25_02_32_31_115.jpg "Recovery Image"
[image5]: ./examples/center_2017_08_25_04_31_48_099.jpg "Recovery Image"
[image6]: ./examples/center_2017_08_25_04_32_33_082.jpg "Normal Image"
[image7]: ./examples/center_2017_08_25_04_32_33_082_flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.ipython block 5) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (block 5). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.ipynb block 5). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code block 6). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.ipynb block 5).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, recovering from water and dirt borders  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA deep learning network for self-driving cars (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because it was created for the specific purpose of self driving cars in mind.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it includes a dropout layer right before th final dense layer.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the water borders and dirt road bordders. To improve the driving behavior in these cases, I ran the car 5-6 times by recording the recovery path only and not the path leading into these borders.
I tried with diferrent other image processing techniques like YUV color space, gamma correction. At the end, the only technique that worked was increasing the brightness in HSV space. I also removed small steering angle changes below 0.015 to balance teh data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and at the center of the road.

####2. Final Model Architecture

The final model architecture (model.ipynb block 5) consisted of a convolution neural network with the following layers and layer sizes -

Input Size: (160,320,3) normalized around mean as x/127.5 - 1.
Cropping: ((70,25),(100,100))
Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
Flatten
Fully connected: neurons: 100, activation: RELU
Fully connected: neurons: 50, activation: RELU
Fully connected: neurons: 10, activation: RELU
Drop out (0.8)
Fully connected: neurons: 1 (Output)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric). Please note, this does not have the drop-out layer I added.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would emulate driving in the opposite direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also augmented the data by using the left and right camera images with a correction of 0.25 on the steering angle.

After the collection process, I had 11707 number of data points.


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by reducing traing and validation erors. I used an adam optimizer so that manually training the learning rate wasn't necessary.
