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
import cv2 as cv

def process_function_variance(result_image,inpaint_mask,start_row,numrows):
    files = glob.glob('testset/skating/*.jpg')

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
        mask_row_value = []
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

            most_pop_centroid = stats.mode(kmeans.labels_)[0][0]
            if index != most_pop_centroid and stats.mode(kmeans.labels_)[1][0] >= len(image_stack)*3/4:
                mask_row_value.append([255,255,255])
            else:
                mask_row_value.append([0,0,0])
            row_value.append( kmeans.cluster_centers_[index].astype(int) )
        result_image[row] = row_value.copy()
        inpaint_mask[row] = mask_row_value.copy()

def main():
    procs = []
    manager = multiprocessing.Manager()
    im = imageio.imread('testset/skating/in000001.jpg',format='jpg')
    result_image = manager.list(np.zeros_like(im))
    inpaint_mask = manager.list(np.zeros_like(im))
    work_per_process = int(im.shape[0]/4)
    for p_num in range(0,4):
        process = multiprocessing.Process(target=process_function_variance, args=(result_image,inpaint_mask,p_num*work_per_process,work_per_process))
        procs.append(process)
        process.start()
    for p in procs:
        p.join()

    #result_image = imageio.imread('kmeans_run.jpg',format='jpg')
    imageio.imsave("kmeans_inpaintmask.png",inpaint_mask)
    imageio.imsave("kmeans_run.png",result_image)
    #plt.imshow(result_image)
    #plt.show()
    # Open the image.
    img = cv.imread('kmeans_run.png')

    # Load the mask.
    mask = cv.imread('kmeans_inpaintmask.png', 0)

    # Inpaint.
    dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

    # Write the output.
    cv.imwrite('kmeans_inpainted.png', dst)


if __name__ == '__main__':
    main()