import os
import random
import cv2
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.utils import np_utils
import numpy as np
import pandas as pd
import pickle

from keras_preprocessing.image import ImageDataGenerator
from skimage.color import rgb2hed
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set the image size for the images
IMG_SIZE = 255

# An excel file that contains the images name and the label of them
train_images_label = 'D:\\train.csv'
# From where to retrieve the images
train_images = 'D:\\QuPathProject\\export\\'
# Convert the csv file into pandas dataframe
df = pd.read_csv(train_images_label)
# Choose only the two columns that are the most important
df = df[['image_id', 'isup_grade']]
# Converting the pandas dataframe into a python dictionary
list_train_label = df.set_index('image_id')['isup_grade'].to_dict()
train_data = []


# This method group the images into classes.
def img_extraction():
    # Each list will contain the images with the correspond category.
    res_0 = []
    res_1 = []
    res_2 = []
    res_3 = []
    res_4 = []
    res_5 = []
    for root, dirs, filename in os.walk(train_images):
        for name in filename:
            if name.endswith('.jpg'):
                if list_train_label[name.split('_')[0]] == 0:
                    res_0.append(name)
                if list_train_label[name.split('_')[0]] == 1:
                    res_1.append(name)
                if list_train_label[name.split('_')[0]] == 2:
                    res_2.append(name)
                if list_train_label[name.split('_')[0]] == 3:
                    res_3.append(name)
                if list_train_label[name.split('_')[0]] == 4:
                    res_4.append(name)
                if list_train_label[name.split('_')[0]] == 5:
                    res_5.append(name)

    # Group all lists into one for further process.
    list_of_arrays = [res_0, res_1, res_2, res_3, res_4, res_5]
    return list_of_arrays


# This method retrieve the images and
def create_images():
    list_of_arrays = img_extraction()

    random_train_image = []
    file_name_count = 0

    for img_array in list_of_arrays:
        for img_name in img_array:
            # Reads each image the list of categories
            image = cv2.imread(train_images + img_name.split('_')[0] + '\\' + img_name)
            # Checks if the image is greater than 256 by 256 pixels
            if IMG_SIZE < image.shape[0] and IMG_SIZE < image.shape[1]:
                random_train_image.append(img_name)
        # After passing the condition of the dimension. The code chooses random 2000 images from the same category
        random_files = np.random.choice(random_train_image, 2000, replace=False)

        for name in random_files:
            # From the patient name choose the category which corresponds
            name_class = list_train_label[str(name).split('_')[0]]
            name_file = name.split('_')[0]
            # Read each image from the 2000 randomly picked
            img_array = cv2.imread(train_images + name_file + '\\' + name)
            # Resized them into the given size
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_data.append([img_array, name_class])

        # Save the data into pickle format (.pkl)
        save_the_data(train_data, file_name_count)
        random_train_image.clear()
        train_data.clear()
        file_name_count += 1


# The method that saves the data into .pkl for each category of data set
def save_the_data(data, filename):
    fileX = open('D:\\QuPathProject\\X_y_without' + str(filename) + '.pkl', 'wb')
    pickle.dump(data, fileX)
    print("The file exported : ", filename)
    fileX.close()


# The method gives the input images and the labels
def creating_features_labels():
    X = []
    y = []

    # Retrieve each pickle data
    data0 = pickle.load(open('/content/drive/My Drive/Data_kaggle/X_y_0.pkl', "rb"))
    data1 = pickle.load(open('/content/drive/My Drive/Data_kaggle/X_y_1.pkl', "rb"))
    data2 = pickle.load(open('/content/drive/My Drive/Data_kaggle/X_y_2.pkl', "rb"))
    data3 = pickle.load(open('/content/drive/My Drive/Data_kaggle/X_y_3.pkl', "rb"))
    data4 = pickle.load(open('/content/drive/My Drive/Data_kaggle/X_y_4.pkl', "rb"))
    data5 = pickle.load(open('/content/drive/My Drive/Data_kaggle/X_y_5.pkl', "rb"))

    # Concat all the data set into one
    data = np.concatenate((data0, data1), axis=0)
    data = np.concatenate((data, data2), axis=0)
    data = np.concatenate((data, data3), axis=0)
    data = np.concatenate((data, data4), axis=0)
    data = np.concatenate((data, data5), axis=0)

    # Shuffling the data set
    random.shuffle(data)

    # Splitting the data set into images and labels
    for features, label in data:
        # The images
        X.append(features)
        # The labels
        y.append(label)

    # Reshaping the image data set into [number of images, width size, height size, channel] in order to feed the
    # neural network
    X = np.asarray(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    y = np.array(y, dtype=np.float32)
    y = np_utils.to_categorical(y)

    return X, y


# Define the Deep Neural Network model
def DNN_Model():
    create_images()

    X, y = creating_features_labels()

    # Data augmentation, in order to generate more images related to our data set
    datagen_train_data = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.2, 0.5),
        shear_range=0.0,
        zoom_range=(0.5, 1),
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None,
    )

    # The data is spited into 90% training and 10% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    datagen_train_data.fit(X_train)

    print("The shape of train data set")
    print(X_train.shape)
    print(y_train.shape)

    print("----------------------------")

    print("The shape of test data set")
    print(X_test.shape)
    print(y_test.shape)

    # Load the ResNet50 model without the last layers (the classification part) and with different input shape
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])

    # We stop all layers of being trained
    for layer in resnet_model.layers:
        layer.trainable = False

    x = resnet_model.output
    x = GlobalAveragePooling2D()(x)

    # We add one single layer for the classification part. We can add as much as we want.
    x = Dense(512, activation='relu')(x)  # dense layer 1
    x = Dropout(0.25)(x)

    # Add to the model the number of classes that we need to classify
    output = Dense(y.shape[1], activation='softmax')(x)

    # We can define the new model
    model = Model(inputs=resnet_model.input, outputs=output)

    # Shows the structure of the model.
    print(model.summary())

    # How many circles would the model apply
    epochs = 100
    # The learning rate of the model
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay)
    batch_size = 128

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    # Train our model by using data augmentation
    model.fit_generator(
        datagen_train_data.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=12000 // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        validation_steps=batch_size)



if __name__ == '__main__':
    DNN_Model()
