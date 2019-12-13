import numpy as np
import glob
import pylab as plt
import imageio
files = glob.glob('testset/fall/*.jpg')

imagenames_list = []
for f in files:
    imagenames_list.append(f)

read_images = []
for n in range(0,40):
    image_in = imageio.imread(imagenames_list[n],format='jpg')
    read_images.append(image_in)

rows = read_images[0][:,:,0].shape[0]
cols = read_images[0][:,:,0].shape[1]
updated_img = np.zeros_like(read_images[0])
channel = 3

for i in range(rows):
    for j in range(cols):
        for c in range(channel):
                value = np.zeros(len(read_images))
                for k in range(len(read_images)):
                    value[k] = read_images[k][i, j, c]
                updated_img[i, j, c] = np.median(value)

#plt.title('Updated Image using Median Stack Filter with '+str(len(read_images))+' samples')
#plt.imshow(updated_img)
#plt.show()
imageio.imsave("MSF_result.png",updated_img)