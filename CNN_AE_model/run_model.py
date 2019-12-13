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
from keras.models import model_from_json

json_file = open('model_testtex.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_testtex.h5")
print("Loaded model from disk")


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
for n in range(0, 40):
    mask_in = imageio.imread(mask_imagenames_list[n])
    image_in = cv.imread(imagenames_list[n],0)
    next_image_in = cv.imread(imagenames_list[n+1],0)
    diff = np.subtract(next_image_in,image_in)
    #entry = np.pad(diff, [(0, 540-tmp_im.shape[0]), (0, 540-tmp_im.shape[1])], mode='constant')
    #mask_in = np.pad(mask_in, [(0, 540-tmp_im.shape[0]), (0, 540-tmp_im.shape[1])], mode='constant')
    train_images.append(diff)
    test_images.append(mask_in)
    #print(np.pad(mask_in, [(0, 540-tmp_im.shape[0]), (0, 540-tmp_im.shape[1])], mode='constant').shape)

train_images = np.array(train_images)
test_images = np.array(test_images)
train_images = train_images.reshape(-1,480,720, 1)
test_images = test_images.reshape(-1, 480,720, 1)
prediction = loaded_model.predict(train_images)
# you can now display an image to see it is reconstructed well
#print(prediction[39])
for imageno in range(0,len(prediction)):
    x =prediction[imageno].reshape(480,720) * 255
    cv.imwrite("output/modelpred" + str(imageno) +  ".png",x)

