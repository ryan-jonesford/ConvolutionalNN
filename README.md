# ConvolutionalNN
A
# **Traffic Sign Recognition Poject**
A convolutional neural network writen in python and Tensorflow for identifying German road signs

This is a project in [Udacity's Self-Driving Car Nano Degree](http://www.udacity.com/drive)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./writeup_images "class_count.png"
[image1] "0 training sample.png"
[image2] "1 training sample.png"
[image3] "2 training sample.png"
[image4] "3 training sample.png"
[image5] "4 training sample.png"
[image6] "5 training sample.png"
[image7] "6 training sample.png"
[image8] "7 training sample.png"
[image9] "8 training sample.png"
[image10] "9 training sample.png"
[image11] "10 training sample.png"
[image12] "11 training sample.png"
[image13] "12 training sample.png"
[image14] "13 training sample.png"
[image15] "14 training sample.png"
[image16] "15 training sample.png"
[image17] "16 training sample.png"
[image18] "17 training sample.png"
[image19] "18 training sample.png"
[image20] "19 training sample.png"
[image21] "20 training sample.png"
[image22] "21 training sample.png"
[image23] "22 training sample.png"
[image24] "23 training sample.png"
[image25] "24 training sample.png"
[image26] "25 training sample.png"
[image27] "26 training sample.png"
[image28] "27 training sample.png"
[image29] "28 training sample.png"
[image30] "29 training sample.png"
[image31] "30 training sample.png"
[image32] "31 training sample.png"
[image33] "32 training sample.png"
[image34] "33 training sample.png"
[image35] "34 training sample.png"
[image36] "35 training sample.png"
[image37] "36 training sample.png"
[image38] "37 training sample.png"
[image39] "38 training sample.png"
[image40] "39 training sample.png"
[image41] "40 training sample.png"
[image42] "41 training sample.png"
[image43] "42 training sample.png"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ryan-jonesford/ConvolutionalNN/blob/master/CNN-Project.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43 classes

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data breaks up. As is clearly seen there are less than 250 samples for some images and more than 2000 for others. This gives us a good dataset for the samples that we have a lot of but not so good for ones like the 20km/h speed limit signs 

![Number of Training signs in each Class/Label][image0]

Since I'm not German and am unfamiliar with their traffic signs I also printed out a sample of each sign with it's attached label:
![1][image1]
![2][image2]
![3][image3]
![4][image4]
![5][image5]
![6][image6]
![7][image7]
![8][image8]
![9][image9]
![10][image10]
![11][image11]
![12][image12]
![13][image13]
![14][image14]
![15][image15]
![16][image16]
![17][image17]
![18][image18]
![19][image19]
![20][image20]
![21][image21]
![22][image22]
![23][image23]
![24][image24]
![25][image25]
![26][image26]
![27][image27]
![28][image28]
![29][image29]
![30][image30]
![31][image31]
![32][image32]
![33][image33]
![34][image34]
![35][image35]
![36][image36]
![37][image37]
![38][image38]
![39][image39]
![40][image40]
![41][image41]
![42][image42]
![43][image43]




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


