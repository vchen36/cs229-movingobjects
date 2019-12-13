# Import all the required Libraries
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
import imageio
import cv2 as cv
import glob
from pathlib import Path
#ENCODER
inp = Input((480, 720,1))
e = Conv2D(32, (3, 3), activation='relu')(inp)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(64, (3, 3), activation='relu')(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(64, (3, 3), activation='relu')(e)
l = Flatten()(e)
l = Dense(150, activation='sigmoid')(l)
#DECODER
d = Reshape((10,15,1))(l)
d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), strides=2,activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), strides=2,activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), strides=3,activation='relu', padding='same')(d)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

ae = Model(inp,decoded)
ae.summary()


mask_imagenames_list = []
#for filename in Path('results').rglob('*.png'):
#    mask_imagenames_list.append(filename)

files = glob.glob('testset/fall_mask/*.png')
for f in files:
    mask_imagenames_list.append(f)
# input
files = glob.glob('testset/fall/*.jpg')
imagenames_list = []
for f in files:
    imagenames_list.append(f)

tmp_im = imageio.imread(mask_imagenames_list[0])
train_images = []
test_images = []
for n in range(0, 39):
    mask_in = imageio.imread(mask_imagenames_list[n])
    image_in = cv.imread(imagenames_list[n],0)
    next_image_in = cv.imread(imagenames_list[n+1],0)
    diff = np.subtract(next_image_in,image_in)
    #entry = np.pad(diff, [(0, 540-tmp_im.shape[0]), (0, 540-tmp_im.shape[1])], mode='constant')
    #mask_in = np.pad(mask_in, [(0, 540-tmp_im.shape[0]), (0, 540-tmp_im.shape[1])], mode='constant')
    train_images.append(diff)
    test_images.append(mask_in)
    #print(np.pad(mask_in, [(0, 540-tmp_im.shape[0]), (0, 540-tmp_im.shape[1])], mode='constant').shape)

#plt.imshow(test_images[0],cmap='gray', vmin=0, vmax=255)
#plt.show()
#print(train_images[0])
train_images = np.array(train_images)
test_images = np.array(test_images)
train_images = train_images.reshape(-1, 480,720, 1)
test_images = test_images.reshape(-1, 480,720, 1)
print(train_images.shape)
print(test_images.shape)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# compile it using adam optimizer
ae.compile(optimizer="adam", loss="mse")
#Train it by providing training images
ae.fit(train_images, test_images, epochs=60)
#IF you want to save the model
model_json = ae.to_json()
with open("model_testtex.json", "w") as json_file:
    json_file.write(model_json)

ae.save_weights("model_testtex.h5")
print("Saved model")

prediction = ae.predict(train_images)
# you can now display an image to see it is reconstructed well
for i in range(0,len(prediction)):
    x =prediction[i].reshape(480,720) * 255
    cv.imwrite("output/modelpred" + str(i) + ".png", x)