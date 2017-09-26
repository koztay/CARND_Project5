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
cells_per_step = 2
spatial_size = (32, 32)
hist_bins = 32
ystart = 400
ystop = 656
scale = 1.5

for img_path in test_images:
    img = mpimg.imread(img_path)

    out_img, heat_map = hf.find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                     pix_per_cell, cell_per_block, cells_per_step, spatial_size, hist_bins)
    labels = label(heat_map)
    # draw bounding boxes on image
    draw_img = hf.draw_labeled_bboxes(np.copy(img), labels)
    out_images.append(draw_img)
    out_titles.append("")
    out_images.append(heat_map)

fig = plt.figure(figsize=(12, 24))
hf.visualize(fig, 8, 2, out_images, "image_pipeline")

