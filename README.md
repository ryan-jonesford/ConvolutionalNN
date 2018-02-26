# ConvolutionalNN
A convolutional neural network written in python and Tensorflow for identifying German road signs
# **Traffic Sign Recognition Project**

This is a project in [Udacity's Self-Driving Car Nano Degree](http://www.udacity.com/drive)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the soft-max probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./writeup_images/class_count.png
[image1]: ./writeup_images/0_training_sample.png
[image2]: ./writeup_images/1_training_sample.png
[image3]: ./writeup_images/2_training_sample.png
[image4]: ./writeup_images/3_training_sample.png
[image5]: ./writeup_images/4_training_sample.png
[image6]: ./writeup_images/5_training_sample.png
[image7]: ./writeup_images/6_training_sample.png
[image8]: ./writeup_images/7_training_sample.png
[image9]: ./writeup_images/8_training_sample.png
[image10]: ./writeup_images/9_training_sample.png
[image11]: ./writeup_images/10_training_sample.png
[image12]: ./writeup_images/11_training_sample.png
[image13]: ./writeup_images/12_training_sample.png
[image14]: ./writeup_images/13_training_sample.png
[image15]: ./writeup_images/14_training_sample.png
[image16]: ./writeup_images/15_training_sample.png
[image17]: ./writeup_images/16_training_sample.png
[image18]: ./writeup_images/17_training_sample.png
[image19]: ./writeup_images/18_training_sample.png
[image20]: ./writeup_images/19_training_sample.png
[image21]: ./writeup_images/20_training_sample.png
[image22]: ./writeup_images/21_training_sample.png
[image23]: ./writeup_images/22_training_sample.png
[image24]: ./writeup_images/23_training_sample.png
[image25]: ./writeup_images/24_training_sample.png
[image26]: ./writeup_images/25_training_sample.png
[image27]: ./writeup_images/26_training_sample.png
[image28]: ./writeup_images/27_training_sample.png
[image29]: ./writeup_images/28_training_sample.png
[image30]: ./writeup_images/29_training_sample.png
[image31]: ./writeup_images/30_training_sample.png
[image32]: ./writeup_images/31_training_sample.png
[image33]: ./writeup_images/32_training_sample.png
[image34]: ./writeup_images/33_training_sample.png
[image35]: ./writeup_images/34_training_sample.png
[image36]: ./writeup_images/35_training_sample.png
[image37]: ./writeup_images/36_training_sample.png
[image38]: ./writeup_images/37_training_sample.png
[image39]: ./writeup_images/38_training_sample.png
[image40]: ./writeup_images/39_training_sample.png
[image41]: ./writeup_images/40_training_sample.png
[image42]: ./writeup_images/41_training_sample.png
[image43]: ./writeup_images/42_training_sample.png
[image44]: ./writeup_images/math.PNG
[image45]: ./writeup_images/normal_gray.png
[image46]: ./signs_from_google_maps/30kph.bmp
[image47]: ./signs_from_google_maps/bicycles_only.bmp
[image48]: ./signs_from_google_maps/childrenxing.bmp
[image49]: ./signs_from_google_maps/donotenter.bmp
[image50]: ./signs_from_google_maps/mandatory_left.bmp
[image51]: ./signs_from_google_maps/no_entry.bmp
[image52]: ./signs_from_google_maps/yield.bmp
[image53]: ./writeup_images/p0Yield.png
[image54]: ./writeup_images/p1No_entry.png
[image55]: ./writeup_images/p2No_entry.png
[image56]: ./writeup_images/p3Speed_limit_30kmph.png
[image57]: ./writeup_images/p4Children_crossing.png
[image58]: ./writeup_images/p5Turn_left_ahead.png
[image59]: ./writeup_images/p6Bicycles_crossing.png
[image60]: ./writeup_images/model_arch.png




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ryan-jonesford/ConvolutionalNN/blob/StandOut)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of_training_set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43 classes

#### 2. Include an exploratory visualization of the data set.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data breaks up. As is clearly seen there are less than 250 samples for some images and more than 2000 for others. This gives us a good data set for the samples that we have a lot of but not so good for ones like the 20km/h speed limit signs 

![Number of_training_signs in each Class/Label][image0]

Since I'm not German and am unfamiliar with their traffic signs I also printed out a sample of each sign with it's attached label:

![1][image1] ![2][image2]![3][image3] ![4][image4] ![5][image5] ![6][image6]
![7][image7] ![8][image8] ![9][image9] ![10][image10] ![11][image11]
![12][image12] ![13][image13] ![14][image14] ![15][image15] ![16][image16]
![17][image17] ![18][image18] ![19][image19] ![20][image20] ![21][image21]
![22][image22] ![23][image23] ![24][image24] ![25][image25] ![26][image26]
![27][image27] ![28][image28] ![29][image29] ![30][image30] ![31][image31]
![32][image32] ![33][image33] ![34][image34] ![35][image35] ![36][image36]
![37][image37] ![38][image38] ![39][image39] ![40][image40] ![41][image41]
![42][image42] ![43][image43]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

Preprocessing the images is an important step in developing a convolutional neural network.  Images with a lot of depth can make the network work harder than needed if that depth isn't important to the identification of the image; as is the case with this project. 

After trying multiple image manipulation techniques i settled on converting the image to grayscale to reduce the depth to a single color channel and then normalizing those images using the formula:

![Normalization Math][image44]

The normalization function is good for producing a near zero mean and variance in the image. 

The above images are from the data set, here is a sample of the image after putting it in grayscale and normalizing it:

![Normalized image][image45]

I also augmented my dataset with modified images of the original dataset. I augmented it in two ways. 

The first way I augmented it was I made a copy of the entire dataset and ran a random size distortion on it. This will help to account for weird camera angles that makes the signs look squished, and smaller images where the signs don’t take up the entire image space. 

The second way that I augmented the dataset was to add more images from test sets that had less. For example I doubled the amount of speed limit 20km/h signs. Those signs that I added, I also ran a random distortion on with equal parts likely to either add noise (jitter) to the image, rotate it, and randomly brighten it. This keeps the augmented dataset fresh, so the model isn’t training on the same images. 

#### 2. Describe what your final model architecture looks like.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution1 3x3     	| 1x1 stride, valid padding, outputs 30x30x6    |
| RELU					|												|
| Convolution2 3x3     	| 1x1 stride, valid padding, outputs 28x28x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution3 3x3     	| 1x1 stride, valid padding, outputs 12x12x32   |
| RELU					|												|
| Convolution4 3x3     	| 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Convolution5 3x3     	| 1x1 stride, valid padding, outputs 3x3x128    |
| RELU					|												|
| Convolution6 3x3     	| 1x1 stride, valid padding, outputs 2x2x140    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 1x1x140 				    |
| Fully connected		| input: 140 output: 120 					    |
| Fully connected		| input: 120 output: 84 						|
| Dropout		  	|  50% 						    |
| Fully connected   	| input: 84 output: 43						    |
| Softmax   			|												|
|						|												|
 
 I used Tensorboard as part of this project and this is the graph produced:

 ![graph][image60]

It clearly shows the 6 convolutional layers, the ones colored green have the max pooling contained in them. The third fully connected layer I have labeled "Logits"


#### 3. Describe how you trained your model.

To train the model, I used the [Adam Optimizer](https://arxiv.org/abs/1412.6980) with a learning rate of 0.001, a batch size of 256, and 50 epochs

I also started out with truncated normal weights with a stdev of .1 and mean of 0. Used valid padding and strides of 1 for my convolutional layers. A stride and kernel size of 2 for my pooling. And finally a 50% dropout rate on my last fully connected layer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* validation set accuracy of about 95% 
* test set accuracy of .918

The fact that my test set was so much lower than my validation sets seems to point to over-fitting. 

I first started out with the simple LeNet architecture since it was the one we used in class. Doing some research on my own I found that there have been many successful architectures that I could choose from, but that’s not exactly what I wanted to do. I wanted to get a feel for how many of the architectures are made and create my own.
I found that many architectures used multiple convolutional layers before pooling and finally running it through a fully connected layer at the end. 
I struggled for many days trying to get a network to run correctly. Eventually I settled on the one that is in my report. There are many hyper-parameters that I had that I can tune and play around with:

* learning rate
* initialization weights and biases
* layer sizes
* dropout rate
* stride length
* kernel size
* padding style

Unfortunately for me, the model that I developed doesn’t seem to be much better than the LeNet model that I started out with. Perhaps if I had played with the hyperparamters a little more and augmented my data before going in search of a different architecture I could have gotten near the same results. 


### Test a Model on New Images

#### 1. Run the model on five German traffic signs found on the web.

While the rubric asked for five I wanted a slightly larger dataset so, here are seven German traffic signs that I found while using Google Street view in a couple different cities in Germany:

![30 Km/h][image46] ![bicycles only][image47] ![children crossing][image48] 
![Do not enter][image49] ![mandatory left ahead][image50]
![do not enter][image51] ![yield][image52]

I expect that all but two of these images should be easy to classify. Image 3 (children crossing) is quite distorted and blurry and the number of examples in the training dataset wasn't very high, so it will be hard to classify correctly.
Image 2 is not a sign that has been classified in the training program. My hope is, is that the classification will be a sign that is similar to it: “bicycle crossing”. The sign actually means “bicycles only”. 

#### 2 & 3. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Describe how certain the model is when predicting on each of the five new images by looking at the soft-max probabilities for each prediction. 


# Here are the results of the prediction:

| Sign              |  Probability  |     Prediction        | Correct?  |
|:-----------------:|:-------------:|:---------------------:|:---------:|
|Yield              | 1.0         	| Yield                 | Yes       |
|No Entry           | 1.0   		| No Entry              | Yes       |
|No Entry           | 1.0			| No Entry              | Yes       |
|Speed limit 30km/h | .99	      	| Speed limit 30km/h    | Yes       |
|Children Crossing   | 1.0			| Right-of-way at next  | No        |
|Turn left Ahead    | 1.0			| Turn left ahead       | Yes       |
|Bicycles Only       | .83			| No Entry           | No        |

The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71.4%.

![yield prediction][image53]
![no entry prediction][image54]
![no entry prediction(2)][image55]
![Speed limit 30km/h prediction][image56]
![Children Crossing prediction][image57]
![Turn left ahead prediction][image53]
![Bicycles crossing prediction][image59]


As can be seen from the graphs above, the model seems to be certain of itself on all the new images I fed it, even though it got two wrong. The only one that it showed some uncertainty on is the one that wasn’t in the dataset. It’s interesting that it interpreted it as “No Entry”; since that would work in a real life scenario (since bicycles only really means “no cars” which could also mean “no entry”), but this feels (and probably is) more like luck. 
