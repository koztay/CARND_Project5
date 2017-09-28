import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
import helper_functions as hf

test_images = glob.glob("test_images/*.jpg")
out_images = []
out_titles = []
out_maps = []
dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
cells_per_step = 2  # this means 75% overlap
spatial_size = (32, 32)
hist_bins = 32
ystart = 400
ystop = 656
scales = [1, 2]  # this means 128x128 window size

heatmaps_collection = []


for img_path in test_images:
    img = mpimg.imread(img_path)

    for scale in scales:
        out_img, heat_map = hf.find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                         pix_per_cell, cell_per_block, cells_per_step, spatial_size, hist_bins)
        heatmaps_collection.append(heat_map)
    sum_of_heatmaps = sum(heatmaps_collection[-2:])  # sum of last X
    # heatmap = hf.apply_threshold(sum_of_heatmaps, threshold=2)  # thresholded heat_maps
    average_heat_map = sum_of_heatmaps / len(heatmaps_collection)

    # heat_sum_average = sum(heat_maps[-2:]) / 2
    # do threshold and some kind of detection (blob or cv.contours)
    # heatmap = hf.apply_threshold(heatmap, 1)
    heatmap = np.clip(average_heat_map, 0, 255)

    labels = label(heatmap)
    # draw bounding boxes on image
    draw_img = hf.draw_labeled_bboxes(np.copy(img), labels)
    out_images.append(draw_img)
    out_titles.append("")
    out_images.append(heatmap)

fig = plt.figure(figsize=(12, 18))
hf.visualize(fig, 6, 2, out_images, title="")
plt.show()
fig.savefig("output_images/05_images_pipeline_combined_scales.png")
