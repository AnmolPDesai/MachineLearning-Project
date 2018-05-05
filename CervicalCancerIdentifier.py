## !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:05:44 2018

@author: anmoldesai
"""

#Importing Relevant Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from shutil import copyfile
import tensorflow
from PIL import Image
import cv2
import numpy as np
import seaborn as sns

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.models import load_model
from keras import losses

img_list = []
def populate_img_list(indir, photo_list): #http://stackoverflow.com/questions/11801309/how-to-loop-over-files-with-python/11801336
    image_count = 0
    error_count = 0
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            if f != ".DS_Store":
                try:
                    img = imread(os.path.join(root, f))
                    photo_list.append([f, indir, img.shape[0], img.shape[1], img.shape[2]])
                    image_count += 1
                except ValueError:
                    error_count += 1
   # print "indir: " + indir
    #print "images: "+ imagecount
    print ("indir: %s images: %s errors: %s" % (indir, image_count, error_count))
    
populate_img_list("/Users/anmoldesai/Desktop/MachineLearning/dataDir/Training Data/train/Type_1", img_list)
populate_img_list("/Users/anmoldesai/Desktop/MachineLearning/dataDir/Training Data/train/Type_2", img_list)
populate_img_list("/Users/anmoldesai/Desktop/MachineLearning/dataDir/Training Data/train/Type_3", img_list)
df = pd.DataFrame(img_list, columns = ["img_name", "path", "height", "width", "chan"])
df["label"] = df.path.str[-6:]
df.head()

#df.describe()
#df.label.value_counts().plot(kind = "bar", title = "Cervix Types")
#df.label.value_counts()/len(df)

############################
type_1_image = df[df.label == "Type_1"].iloc[0].path + "/" + df[df.label == "Type_1"].iloc[0].img_name
print ("Type 1")
#Image(type_1_image, width = 200, height = 200)
############################

#############################
#Train Test Split#
X_train, X_test, y_train, y_test = train_test_split(df[["path", "img_name", "label"]], df["label"], test_size=0.3, stratify = df["label"], random_state = 754142)
#############################

def copy_files(dataframe, destination_folder):
    if not os.path.exists(destination_folder): #Assumption is if path exists, then images are in that folder
        os.makedirs(destination_folder)
        os.makedirs(destination_folder + "/Type_1")
        os.makedirs(destination_folder + "/Type_2")
        os.makedirs(destination_folder + "/Type_3")
        for index, image in dataframe.iterrows():
            source_path = image["path"] + "/" + image["img_name"]
            dst_path = destination_folder + "/" + image["label"] + "/" + image["img_name"]
            copyfile(source_path, dst_path)
            
copy_files(X_train, "train_split")
copy_files(X_test, "val_split")

####Building the Model####
model = Sequential()
#Conv Set 1
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150), data_format = "channels_first"))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Conv Set 2
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Conv Set 3
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2))) (Keras would not let me pool once more due to dimensions)

#FC Set 1 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

#FC Set 2
model.add(Dense(128))
model.add(Activation('relu'))

#Final FC
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss=tensorflow.losses.log_loss,
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

batch_size = 300

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, data_format = "channels_first")

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255, data_format = "channels_first")

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'train_split',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        'val_split',
        target_size=(150, 150),
        batch_size=batch_size)

history = model.fit_generator(
        train_generator,
        validation_data = validation_generator,
        steps_per_epoch=(2000 // batch_size),
        epochs=100,
        validation_steps=(800 // batch_size))

model.save("my_first_neural_network_20170513.h5")

final_test_datagen = ImageDataGenerator(rescale=1./255, data_format = "channels_first")

final_test_generator = final_test_datagen.flow_from_directory(
        '/Users/anmoldesai/Desktop/MachineLearning/dataDir/TestData',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode = None)

predictions = model.predict_generator(
        final_test_generator,
        steps = (2000 // batch_size))

submission_list = []
for f, prediction in zip(final_test_generator.filenames, predictions):
    submission_list.append([f.strip("test/"), prediction[0], prediction[1], prediction[2]])
submission_frame = pd.DataFrame(submission_list, columns = ["image_name","Type_1","Type_2","Type_3"])
submission_frame.to_csv("/Users/anmoldesai/Desktop/MachineLearning/Code/Network.csv", index = False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("submissions/my_first_neural_network_20170513.png")
plt.show()

plt.plot(history.history['categorical_accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("submissions/my_first_neural_network_20170513.png")
plt.show()