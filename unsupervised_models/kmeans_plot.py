import imageio
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import pylab as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
from  sklearn.metrics import pairwise_distances_argmin_min
files = glob.glob('input/*.jpg')

imagenames_list = []
for f in files:
    imagenames_list.append(f)

image_stack = []
for n in range(0,40):
    image_in = imageio.imread(imagenames_list[n],format='jpg')
    image_stack.append(image_in)

row_num = image_stack[0].shape[0]
col_num = image_stack[0].shape[1]

row = 241
col = 434
pixel_rgb_list = []
for t in range(0,len(image_stack)):
    pixel_rgb_list.append(image_stack[t][row][col])
pixel_rgb_list = np.array(pixel_rgb_list)
kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_rgb_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(pixel_rgb_list)
ax.scatter(pixel_rgb_list[:,0], pixel_rgb_list[:,1], pixel_rgb_list[:,2],c=kmeans.labels_.astype(float))
plt.show()

