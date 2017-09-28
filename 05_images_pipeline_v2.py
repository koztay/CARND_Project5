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
print("pix_per_cell :", pix_per_cell)
cell_per_block = dist_pickle["cell_per_block"]
print("cell_per_block:", cell_per_block)
color_space = "YCrCb"
spatial_size = (32, 32)
hist_bins = 32
hog_channel = "ALL"
spatial_feat = True
hist_feat = True
hog_feat = True


test_images = glob.glob("test_images/*.jpg")
images = []
titles = []
y_start_stop = [400, 656]
overlap = 0.75
xy_window = (128, 128)

"""
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
"""

for img_path in test_images:
    t1 = time.time()
    img = mpimg.imread(img_path)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255  # normalize img (we trained with pngs, and now reading jpgs.
    # print(np.min(img), np.max(img))  # make sure that image is normalized

    windows = hf.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                              xy_window=xy_window, xy_overlap=(overlap, overlap))

    hot_windows = hf.search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    image_heat = np.zeros_like(img[:, :, 0])
    for window in windows:
        print(window)
        top_left = window[0]
        bottom_right = window[1]
        image_heat[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] += 1

    for window in windows:  # it is like box coordinates
        if np.max(image_heat[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]) > 1:
            image_heat[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] += 2

    window_img = hf.draw_boxes(draw_img, hot_windows, color=(0, 255), thick=6)
    images.append(window_img)
    titles.append("")
    print(time.time() - t1, "seconds to process single image search", len(windows), "windows")

    labels = label(image_heat)
    # draw bounding boxes on image
    draw_img = hf.draw_labeled_bboxes(np.copy(img), labels)

    titles.append("")
    images.append(image_heat)

fig = plt.figure(figsize=(12, 18))
hf.visualize(fig, 6, 2, images, title="")
plt.show()
fig.savefig("output_images/05_images_pipeline_combined_scales_v2.png")
