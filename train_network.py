#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 00:45:32 2018

@author: jayneel
"""

import matplotlib
matplotlib.use("Agg")
 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
 
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
            
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
 
		model.add(Dense(classes))
		model.add(Activation("softmax"))
 
		return model
            
    
#dataset= "images_train"
dataset="ab/output"    
model123= "inno.model"
plot="plot.png"

EPOCHS = 25
INIT_LR = 1e-3
BS = 32
 
print("[INFO] loading images...")
data = []
labels = []
 
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-1]
    #print(label)
    """
    if label.startswith("overpass") :
        label1 = 1 
    if label.startswith("storagetank") : 
        label1 = 2 
    if label.startswith("chaparral") :
        label1 = 3 
    if label.startswith("beach") :
        label1 = 4 
    if label.startswith("freeway") :
        label1 = 5 
    if label.startswith("parkinglot") :
        label1 = 6 
    if label.startswith("agriculture")  :
        label1 = 7 
    if label.startswith("airplane") :
        label1 = 8
    if label.startswith("golfcourse") :
        label1 = 9 """
    if label.startswith("aa_original_overpass") :
        label1 = 1 
    if label.startswith("aa_original_storagetank") : 
        label1 = 2 
    if label.startswith("aa_original_chaparral") :
        label1 = 3 
    if label.startswith("aa_original_beach") :
        label1 = 4 
    if label.startswith("aa_original_freeway") :
        label1 = 5 
    if label.startswith("aa_original_parkinglot") :
        label1 = 6 
    if label.startswith("aa_original_agriculture")  :
        label1 = 7 
    if label.startswith("aa_original_airplane") :
        label1 = 8
    if label.startswith("aa_original_golfcourse") :
        label1 = 9    
    labels.append(label1)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#print(labels)
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
 
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
# initialize the model
print("[INFO] compiling model...")
model =build(width=28, height=28, depth=3, classes=10)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
print("[INFO] serializing network...")
model.save(model123)
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy ")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot)