# Kaggle_Challenge

This repo demonstrates a Kaggle challenge ragarding cancer prostate [link](https://www.kaggle.com/c/prostate-cancer-grade-assessment). In this challenge has purpose to predict each patient the grade of the cancer. In order to achieved that, I created a transfer learning model by using ResNet50. However, image pre-processing is needed first, before "feeding" the neural network. 


### Data set

The data set is constructed through 10.616 images which each image corrensponds to a single patient. The images are in **.TIFF** format. Each image belongs to ISUP grade. That grade demonstrates the cancer grade. 

<p align="center"> 
<img src="https://github.com/BardisRenos/Kaggle_Challenge/blob/master/img.JPG" width="450" height="250" style=centerme>
</p>

### Data process 

This block shows how should the images be processed. Each image is in **.TIFF** format which consists 3 different dimension images of the same pantient. In order to process the images to work with a deep learning model, it is necessary to retrieve the median image out the 3 ones. Another worth mentioning is that to split each image into small tiles and converting them into **.JPG**. Namely, each cancer image is splited into many more small pieces which are focussed mainly on the cancer part. After creating N number of tiles there is a need to exclude the images that are below of a dimension threshold. The remaining tiles are reshaped into same dimension. 

### Choosing Deep Learing Model

To train the dataset of images I choose a pre trained model. That has prove that produce high percentage of accuracy. Therefore **ResNet50** is that I choose. It has already trained weights for a different challenge. However, the tranfer learning has proved to produced higher accuracy comparing to a model that is trained from scratch. 

### Description

ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224. 

<p align="center"> 
<img src= "https://github.com/BardisRenos/Kaggle_Challenge/blob/master/ResNet.png" width="450" height="250" style=centerme>
</p>


### Applying M.I.L
In our case the model that we have applied, is a **Multiple Instance Learning** Convolutional Neural Network. That means each patient's image is divided into small pieces (tiles). By doing that the model is feeded with more images by each category. As can been shown below. Each patient's image is devided into smaller pieces in order to feed the Neural Network with different sides of the same category tissue .

<img src= "https://github.com/BardisRenos/Kaggle_Challenge/blob/master/0a6c5a120961974a7dae8cf11245ff73_Image122.jpg" width="250"/> <img src= "https://github.com/BardisRenos/Kaggle_Challenge/blob/master/0a6c5a120961974a7dae8cf11245ff73_Image23.jpg" width="250"/> <img src= "https://github.com/BardisRenos/Kaggle_Challenge/blob/master/0a6c5a120961974a7dae8cf11245ff73_Image98.jpg" height="250"/>


### Choosing technologies

To develop this challenge by using **Python3** as programing language, **Keras** for the front end and **Tensorflow** for the backed end of the Deep Learning model. 

### Applying image augmentation 

Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset.
Training deep learning neural network models on more data can result in more skillful models, and the augmentation techniques can create variations of the images that can improve the ability of the fit models to generalize what they have learned to new images.
The Keras deep learning neural network library provides the capability to fit models using image data augmentation via the ImageDataGenerator class.

<img src= "https://github.com/BardisRenos/Kaggle_Challenge/blob/master/imageAug.jpg" width="250"/> <img src= "https://github.com/BardisRenos/Kaggle_Challenge/blob/master/imageAug.png" width="250"/>

### BatchNormalization layer

Batch normalization is one of the reasons why deep learning has made such outstanding progress in recent years. 
Batch normalization enables the use of higher learning rates, greatly accelerating the learning process.
It also enabled the training of deep neural networks with sigmoid activations that were previously deemed too difficult to train due to the vanishing gradient problem. Based on its success, other normalization methods such as layer normalization and weight normalization have appeared and are also finding use within the field.


### Changing to the pre trained model attributes

In order to train our dataset and achieve high accuracy score, this approach uses the **ResNet50** deep learning model. In the code I change the input shape of the model and also the number of the output classes. That way the model is adapted into our data set shape. Also the model has one layer of the dense full connected layer of 512 neuros with reLu activation function. 




