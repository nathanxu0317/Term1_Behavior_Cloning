# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./IMG_git/center_lane.jpg "Center line"
[image2]: ./IMG_git/recovery1.jpg "Recovery1"
[image3]: ./IMG_git/recovery2.jpg "Recovery2"
[image4]: ./IMG_git/recovery3.jpg "Recovery3"
[image5]: ./IMG_git/flipedbefore.jpg "Before flip"
[image6]: ./IMG_git/flipedafter.jpg "After flip"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model configuration is given as below:

Layer (type)                     Output Shape          Param #     Connected to                     
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
Conv1 (Convolution2D)            (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
Conv2 (Convolution2D)            (None, 20, 77, 36)    21636       Conv1[0][0]                      
Conv3 (Convolution2D)            (None, 8, 37, 48)     43248       Conv2[0][0]                      
Conv4 (Convolution2D)            (None, 6, 35, 64)     27712       Conv3[0][0]                      
Conv5 (Convolution2D)            (None, 4, 33, 64)     36928       Conv4[0][0]                      
dropout_1 (Dropout)              (None, 4, 33, 64)     0           Conv5[0][0]                      
flatten_1 (Flatten)              (None, 8448)          0           dropout_1[0][0]                  
FC1 (Dense)                      (None, 100)           844900      flatten_1[0][0]                  
FC2 (Dense)                      (None, 50)            5050        FC1[0][0]                        
FC3 (Dense)                      (None, 10)            510         FC2[0][0]                        
dense_1 (Dense)                  (None, 1)             11          FC3[0][0]                        

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.
At the same time, for the first 3 Conv2D layer, 'subsample' has been set to (2,2), in order to avoid generate too many parameters.

A large number of test data was used. (more than 50k images were used for model training). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually to 0.0001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Below is a list of data collected:
1. Normal driving, 2 laps
2. Reverse direction driving, 1 lap
3. Off-nominal driving (recovery lap), 1 lap

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reproduce what NVIDIA did in this paper: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

My first step was to use a convolution neural network model similar to the CNN used in that paper. I thought this model might be appropriate because the objective is very similar. In order to prevent overfitting, one dropout layer was added.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had simialr mean squared error on both the training set and validation set. This indicates my model is working quite well. 

The final step was to run the simulator to see how well the car was driving around track one. Initially there were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I reclected more data with better driving, and also added a recovery lap. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture (model.py lines 54-68) consisted of a convolution neural network with details shown in above section. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I had 57870 of data points. I then preprocessed this data by randomly flipt the image left to right,  randomly adjust brightness of the image, and Randomly adjust RGB of the image. I then croped the top side of the image. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by continues decrease of loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.


