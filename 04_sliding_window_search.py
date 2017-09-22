import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
import time
import helper_functions as hf


dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
# # spatial_size = dist_pickle["spatial_size"]
# # hist_bins = dist_pickle["hist_bins"]
# # color_space = dist_pickle["color_space"]
# # print(color_space)
spatial_size = (32, 32)
hist_bins = 32

img = mpimg.imread('test_images/test3.jpg')
images = glob.iglob('test_images/*.jpg')


ystart = 400
ystop = 656
scales = [1.2, 1.4, 1.7, 2.0, 2.3, 2.7]
cells_per_step = 3
min_rect_size = 80*80


def heat_filter_for_detected_cars(img):
    box_list = []
    for scale in scales:
        box_list_loop = hf.find_cars(img,
                                     ystart,
                                     ystop,
                                     scale,
                                     svc,
                                     X_scaler,
                                     orient,
                                     pix_per_cell,
                                     cell_per_block,
                                     cells_per_step,
                                     spatial_size,
                                     hist_bins)

        box_list.extend(box_list_loop)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = hf.add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = hf.apply_threshold(heat, 5)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # print(labels)
    detected_cars_img = hf.draw_labeled_bboxes(np.copy(img), labels, min_rect_size=min_rect_size)
    draw_img = hf.draw_unlabeled_bboxes(np.copy(img), box_list)
    return detected_cars_img, draw_img, heatmap


plottables = []
for index, image_path in enumerate(images):
    t = time.time()
    img = mpimg.imread(image_path)
    detected_cars_img, draw_img, heatmap = heat_filter_for_detected_cars(img)
    plottables.append([detected_cars_img, draw_img, heatmap])
    mpimg.imsave("output_images/detected_cars_img_{}.jpg".format(index), detected_cars_img)
    mpimg.imsave("output_images/draw_img_{}.jpg".format(index), draw_img)
    mpimg.imsave("output_images/heatmap_{}.jpg".format(index), heatmap)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to work on an image...')




