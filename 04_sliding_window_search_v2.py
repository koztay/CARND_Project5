import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
import helper_functions as hf

test_images = glob.glob("test_images/*.jpg")
images = []
titles = []
y_start_stop = [None, None]
overlap = 0.5

svc = xx  # get it from pickle
X_scaler = xx # get it from pickle
color_space = "YCrCb"
spatial_size = (32, 32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "All"
spatial_feat = True
hist_feat = True
hog_feat = True

for img_path in test_images:
    t1 = time.time()
    img = mpimg.imread(img_path)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255  # normalize img (we trained with pngs, and now reading jpgs.
    print(np.min(img), np.max(img))  # make sure that image is normalized

    windows = hf.slide_window(img, xtart_stop=[None, None], y_start_stop=y_start_stop,
                              xy_window=(128, 128), xy_overlap=(overlap, overlap))

    hot_windows = hf.search_window(img, windows, svc, X_scaler, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = hf.draw_boxes(draw_img, hot_windows, color=(0, 255), thick=6)
    images.append(window_img)
    titles.append("")
    print(time.time() - t1, "seconds to process single image search", len(windows), "windoows")

fig = plt.figure(figsize=(12, 18), dpi=300)
hf.visualize(fig, 5, 2, images, titles)
plt.show()
