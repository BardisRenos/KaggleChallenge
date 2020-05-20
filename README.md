# Kaggle_Challenge

This repo demonstrates a Kaggle challenge ragarding cancer prostate [link](https://www.kaggle.com/c/prostate-cancer-grade-assessment). In this challenge has purpose to predict each patient the grade of the cancer. In order to achieved that, I created a transfer learning model by using ResNet50. However, image pre-processing is needed first, before "feeding" the neural network. 



### Data set

The data set is constructed through 10.616 images which each image corrensponds to a single patient. The images are in .TIFF format, which contains three different images with different sizes into one image file. Each image belongs to ISUP grade. That grade demonstrates the cancer grade. 

<p align="center"> 
<img src="https://github.com/BardisRenos/Kaggle_Challenge/blob/master/img.JPG" width="450" height="250" style=centerme>
</p>

### Data process 

This block shows how should the images be process. 
