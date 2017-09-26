import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
import time
import helper_functions as hf

from moviepy.editor import VideoFileClip, clips_array


dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = (32, 32)
hist_bins = 32
ystart = 400
ystop = 656
scales = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
cells_per_step = 2
min_rect_size = 40*40  # use default value for video
max_aspect_ratio = 2.0  # as width / height

heatmaps_collection = []


def heat_filter_for_detected_cars(img, threshold):
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

    # Add  heats to the list and calculate average od them
    heatmaps_collection.append(heat)
    heat_sum = sum(heatmaps_collection[-5:]) / 5

    # Apply threshold to help remove false positives
    heat = hf.apply_threshold(heat_sum, threshold=threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # print(labels)
    detected_cars_img = hf.draw_labeled_bboxes(np.copy(img), labels,
                                               # min_rect_size=min_rect_size,
                                               max_aspect_ratio=max_aspect_ratio)
    draw_img = hf.draw_unlabeled_bboxes(np.copy(img), box_list)
    return detected_cars_img, draw_img, heatmap


class VideoPipeline:

    def __init__(self):
        super().__init__()
        self.box_list = []
        self.frames = []

    def process_image(self, img, return_value=0):
        detected_cars_img, draw_img, heatmap = heat_filter_for_detected_cars(img, threshold=5)
        if return_value == 0:
            return detected_cars_img
        elif return_value == 1:
            return draw_img
        else:
            return heatmap

    def video_pipeline(self, input_video_path, output_video_path):

        # clip1 = VideoFileClip(input_video_path).subclip(40, 50)
        # clip2 = clip1.fl_image(self.process_image).subclip(40, 50)
        clip1 = VideoFileClip(input_video_path).subclip(15, 25)

        # test_clip.fl_image(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YUV))
        clip2 = clip1.fl_image(lambda x: self.process_image(x, return_value=0))
        # clip3 = clip1.fl_image(lambda x: self.process_image(x, return_value=1))
        # clip4 = clip1.fl_image(lambda x: self.process_image(x, return_value=2))
        # clip3 = clip1.fl_image(self.process_image)

        # final_clip = clips_array([[clip1, clip2], [clip3]])

        clip2.write_videofile(output_video_path, audio=False)

        # final_clip = clips_array([[clip1, clip2],
        #                           [clip3, clip4]])
        # final_clip.resize(width=480).write_videofile("my_stack.mp4")

        return True


""" 
        clip1 = VideoFileClip(input_video_path)
        final_clip = clip1.fl_image(self.full_pipe_for_single_image)
        final_clip.write_videofile(output_video_path, audio=False)
"""

video_pipe = VideoPipeline()
video_pipe.video_pipeline(input_video_path="project_video.mp4", output_video_path="output_video/project_video.mp4")