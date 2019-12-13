import scipy.misc
import random
import imageio
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import pylab as plt
import glob
import multiprocessing
from  sklearn.metrics import pairwise_distances_argmin_min
import statistics
def process_function(result_image,start_row,numrows):
    files = glob.glob('testset/fall/*.jpg')

    imagenames_list = []
    for f in files:
        imagenames_list.append(f)

    image_stack = []
    for n in range(0,40):
        image_in = imageio.imread(imagenames_list[n],format='jpg')
        image_stack.append(image_in)

    row_num = image_stack[0].shape[0]
    col_num = image_stack[0].shape[1]

    for row in range(start_row,start_row+numrows):
        print("finised row ",row)
        row_value = []
        for col in range(0,col_num):
            pixel_rgb_list = []
            for t in range(0,len(image_stack)):
                pixel_rgb_list.append(image_stack[t][row][col])
            pixel_rgb_list = np.array(pixel_rgb_list)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_rgb_list)
            most_pop_centroid = stats.mode(kmeans.labels_)
            row_value.append( kmeans.cluster_centers_[most_pop_centroid[0][0]].astype(int) )
        result_image[row] = row_value.copy()



def process_function_mod(result_image,start_row,numrows):
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

    for row in range(start_row,start_row+numrows):
        print("finised row ",row)
        row_value = []
        for col in range(0,col_num):
            pixel_rgb_list = []
            for t in range(0,len(image_stack)):
                pixel_rgb_list.append(image_stack[t][row][col])
            pixel_rgb_list = np.array(pixel_rgb_list)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_rgb_list)
            most_pop_centroid = stats.mode(kmeans.labels_)
            closest,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_,pixel_rgb_list)
            row_value.append( pixel_rgb_list[closest[most_pop_centroid[0][0]]] )
        result_image[row] = row_value.copy()

def process_function_coinflip(result_image,start_row,numrows):
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

    for row in range(start_row,start_row+numrows):
        print("finised row ",row)
        row_value = []
        for col in range(0,col_num):
            pixel_rgb_list = []
            for t in range(0,len(image_stack)):
                pixel_rgb_list.append(image_stack[t][row][col])
            pixel_rgb_list = np.array(pixel_rgb_list)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_rgb_list)
            most_pop_centroid = stats.mode(kmeans.labels_)
            closest,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_,pixel_rgb_list)
            row_value.append( pixel_rgb_list[closest[np.random.randint(0,2)]] )
        result_image[row] = row_value.copy()

def process_function_biasedcoinflip(result_image,start_row,numrows):
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

    for row in range(start_row,start_row+numrows):
        print("finised row ",row)
        row_value = []
        for col in range(0,col_num):
            pixel_rgb_list = []
            for t in range(0,len(image_stack)):
                pixel_rgb_list.append(image_stack[t][row][col])
            pixel_rgb_list = np.array(pixel_rgb_list)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_rgb_list)
            most_pop_centroid = stats.mode(kmeans.labels_)
            closest,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_,pixel_rgb_list)
            coinflip = np.random.randint(0,3)
            if coinflip <= 1:
                index = most_pop_centroid[0][0]
            else:
                index = (most_pop_centroid[0][0] + 1)%2
            row_value.append( pixel_rgb_list[closest[index]] )
        result_image[row] = row_value.copy()

def process_function_variance(result_image,start_row,numrows):
    files = glob.glob('testset/skating/*.jpg')

    imagenames_list = []
    for f in files:
        imagenames_list.append(f)

    image_stack = []
    for n in range(0,20):
        image_in = imageio.imread(imagenames_list[n],format='jpg')
        image_stack.append(image_in)

    row_num = image_stack[0].shape[0]
    col_num = image_stack[0].shape[1]

    for row in range(start_row,start_row+numrows):
        print("finised row ",row)
        row_value = []
        for col in range(0,col_num):
            pixel_rgb_list = []
            for t in range(0,len(image_stack)):
                pixel_rgb_list.append(image_stack[t][row][col])
            pixel_rgb_list = np.array(pixel_rgb_list)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixel_rgb_list)
            groups = [[] for i in range(0,2)]
            for j in range(0,len(pixel_rgb_list)):
                groups[kmeans.labels_[j]].append(pixel_rgb_list[j])
            group0_var = np.var(groups[0])
            group1_var = np.var(groups[1])
            if group0_var < group1_var:
                index = 0
            else:
                index = 1

            row_value.append( kmeans.cluster_centers_[index].astype(int) )
        result_image[row] = row_value.copy()

def main():
    procs = []
    manager = multiprocessing.Manager()
    im = imageio.imread('testset/skating/in000001.jpg',format='jpg')
    result_image = manager.list(np.zeros_like(im))
    work_per_process = int(im.shape[0]/4)
    for p_num in range(0,4):
        process = multiprocessing.Process(target=process_function_variance, args=(result_image,p_num*work_per_process,work_per_process))
        procs.append(process)
        process.start()
    for p in procs:
        p.join()

    imageio.imsave("kmeans_MPC_run.png",result_image)


if __name__ == '__main__':
    main()