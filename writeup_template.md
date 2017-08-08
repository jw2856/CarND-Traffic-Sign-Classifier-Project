# **Traffic Sign Recognition** 

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

[example]: ./examples/example.png "Example"
[training_set_distribution]: ./examples/training_set_distribution.png "Training Set Distribution"
[validation_set_distribution]: ./examples/validation_set_distribution.png "Validation Set Distribution"
[test_set_distribution]: ./examples/test_set_distribution.png "Test Set Distribution"

[image1]: ./examples/german-sign-1-3.png "Traffic Sign 1"
[image2]: ./examples/german-sign-2-13.png "Traffic Sign 2"
[image3]: ./examples/german-sign-3-12.png "Traffic Sign 3"
[image4]: ./examples/german-sign-4-23.png "Traffic Sign 4"
[image5]: ./examples/german-sign-5-31.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

My project code is at located [here](https://github.com/jw2856/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

I used the code from the LeNet lab to display a random image from the dataset and the corresponding y value. Doing this a few times let me verify that I had properly imported the dataset and the index of each image correctly corresponded to the image's label.

#### Example
![alt text][example]

I create a function to plot the distribution of labels in each of the training, validation, and test sets. The charts showed that there were some labels that were represented more heavily than others, with this distribution being roughly the same between each of the data sets.

#### Training Set
![alt text][training_set_distribution]

#### Validation Set
![alt text][validation_set_distribution]

#### Test Set
![alt text][test_set_distribution]

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

Without any optimizations, I started with the following parameters:

```
Epochs: 20
Batch size: 128
Learning rate: 0.001
mu: 0
sigma: 0.05
```

This seemed to give pretty good results, however did not quite get us to the validation accuracy of 93%:

```
EPOCH 20 ...
Training Accuracy = 0.991
Validation Accuracy = 0.924
```

I started by normalizing the data so that the values were between -0.5 and 0.5.

```
EPOCH 20 ...
Training Accuracy = 0.994
Validation Accuracy = 0.898
```

Oddly, the non-normalized result seemed to be better than the normalized result, but in both cases the validation accuracy was quite a bit off from the training accuracy, which hints at overfitting.

I decreased the learning rate to 0.0005, which seemed to cause a regression on both accuracies without bringing the training and validation accuracies closer:

```
EPOCH 20 ...
Training Accuracy = 0.985
Validation Accuracy = 0.873
```

I decided to add two dropout layers after each of the two fully connected layers, and started with a keep probability of 0.5, and returned the learning rate to 0.001. This seemed to help a lot, and helped us get to a validation accuracy of over 93%:

```
EPOCH 20 ...
Training Accuracy = 0.988
Validation Accuracy = 0.940
```

I experimented with keep probabilities of 0.4 and 0.6, but 0.5 seemed to have the best result. I then increased the epochs to 40, which got an incrementally better result:

```
EPOCH 40 ...
Training Accuracy = 0.997
Validation Accuracy = 0.949
```

I then used the same parameters but adapted the preprocessing to include converting the images to grayscale. This didn't seem to yield any additional benefits, with a representative result shown below. The result was pretty close, and slightly inferior (though potentially negligibly) to the color results.

```
EPOCH 40 ...
Training Accuracy = 0.992
Validation Accuracy = 0.941
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My best model results were:
* training set accuracy of 99.7%
* validation set accuracy of 96.0%
* test set accuracy of 94.0%

I started with the suggested LeNet architecture, with parameter settings as indicated above. From the outset, it seemed like the base architecture seemed to work decently, though did not hit the target validation accuracies for this project. When the results started to indicate that overfitting the test set might be occurring, I tried

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

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

I ran the images through a few versions of the model. The accuracy ranged from 60% to 100% on these images.

The hardest images for were the first image and the last image. The first image was a 60 km/hr speed limit sign, which the model had a difficult time distinguishing from other speed limit signs, which look similar except for one of the numbers. Even in the model that got it right, we can see that the model wasn't very sure on the right answer, generating the following top 5 softmax values and label predictions:

```
[ 0.52610296,  0.47328204,  0.00058817,  0.00002117,  0.00000348]
[ 3,  5,  2, 10, 35]
```

The model had only a 52.6% certainty that the sign was of label 3, though it was correct with the final model I chose.

During this part of the exercise, it was also clear that properly fitting the images of the sign to fit within the 32x32 size was important. In earlier passes, the final image was smaller and not well-centered, which made it difficult for the model to predict accurately. Adjusting the image improved the result and we obtained an accurate prediction, though this remained one of the tougher images. The model was able to see the red triangle sign, but wasn't as confident about the inner marking. Perhaps higher resolution images would be helpful here.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)     		| Speed limit (60km/h)  									| 
| Yield     			| Yield										|
| Priority road					| Priority road											|
| Slippery road      		| Slippery road				 				|
| Wild animals crossing			| Wild animals crossing    							|

My final model achieved 100% on the 5 images I chose (although the initial one was at 60%).

#### Image Prediction Probabilities

**Image 1**

Correct image: Speed limit (60km/h) (3)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52610296  			| Speed limit (60km/h) (13)								| 
| .47328204    				| Speed limit (80km/h) (12)										|
| .00058817				| Speed limit (50km/h) (10)									|
| .00002117	| No passing for vehicles over 3.5 metric tons (25)	|
| .00000348				    | Ahead only (38) 							|

**Image 2**

Correct image: Yield (13)

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| 1               | Yield (13)              | 
| 0             | Priority road (12)              |
| 0        | No passing for vehicles over 3.5 metric tons (10)     |
| 0              | Road work (25)          |
| 0           | Keep right (38)             |


**Image 3**

Correct image:  Priority road (12)

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| 1               | Priority road (12)               | 
| 0             | End of no passing by vehicles over 3.5 metric tons (42)   |
| 0        | Yield (13)           |
| 0              | No passing for vehicles over 3.5 metric tons (10) |
| 0           | End of speed limit (80km/h) (6)         |

**Image 4**

Correct image:  Slippery road (23)

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| .99906546             | Slippery road (23)            | 
| .00050164    | Beware of ice/snow (30)     |
| .00035347       | Dangerous curve to the right (20)     |
| .00005738       | Dangerous curve to the left (19) |
| .00001925       | Road work (25)          |

**Image 5**

Correct image:  Wild animals crossing (31)

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| .98902494             | Wild animals crossing (31)            | 
| .01070386    | Double curve (21)     |
| .00025893       | Dangerous curve to the left (19)     |
| .00000981       | Road narrows on the right (24) |
| .00000163       | Slippery road (23)          |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


