import cv2
from collections import deque
import numpy as np
import pickle
from scipy.ndimage.measurements import label
import helper_functions as hf

from moviepy.editor import VideoFileClip


dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = (32, 32)
hist_bins = 32
# ystart = 400
# ystop = 656
# scales = [(1.0, 400, 464), (1.5, 464, 656), (2.0, 400, 656)]
# scales = [(1.0, 400, 496), (2.0, 464, 656)]
scales = [(1.5, 400, 656), (2.0, 400, 656)]
cells_per_step = 2

heatmaps_collection = []
last_good_labels = []


carslist = []
NUM_ITERATIONS_TO_KEEP = 5
KEEP_LAST_STATUS = 10
INTERSECT_RATIO = 0.9


class Vehicle:
    def __init__(self):
        self.status = []  # this is an array of detected and non-detected
        # i will take this arrays last 5 elements and if all of them is True
        # it means detected and all of them is False then it means not detected
        self.last_n_detection = True
        self.vehicle_number = 1
        self.detected = False  # was the vehicle detected in the last iteration
        self.number_of_detections = 0  # Number of times this vehicle has been detected
        self.number_of_non_detections = 0  # Number of times this vehicle has not been detected
        self.xpixels = None
        self.ypixels = None
        self.recent_xfitted = []
        self.bestx = None
        self.recent_yfitted = []
        self.besty = None
        self.recent_wfitted = []
        self.bestw = None
        self.recent_hfitted = []
        self.besth = None
        self.is_new = True


# this is just for using vehicle class
def draw_labeled_bboxes(img, labels, carslist):
    x_intersect_ratio = 0
    y_intersect_ratio = 0

    if labels[1] > 0:  # if detected cars available
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])

                # check the detected heatmap belongs to which car
                # print("car_index: ", car_number-1)
                car = carslist[car_number - 1]

                # if car has xpixels check if there is a match
                if not car.is_new:  # it means it has xpixels and ypixels
                    x_match = False
                    y_match = False

                    unique_x_pixels = np.unique(car.xpixels)
                    intersect_x = np.intersect1d(nonzerox, unique_x_pixels)
                    x_intersect_ratio = len(intersect_x) / len(unique_x_pixels)
                    print(x_intersect_ratio)
                    if x_intersect_ratio > INTERSECT_RATIO:
                        x_match = True
                        # print("x matched")

                    unique_y_pixels = np.unique(car.ypixels)
                    intersect_y = np.intersect1d(nonzeroy, unique_y_pixels)
                    y_intersect_ratio = len(intersect_y) / len(unique_y_pixels)
                    print(y_intersect_ratio)
                    if y_intersect_ratio > INTERSECT_RATIO:
                        y_match = True
                        # print("y matched")

                    if x_match and y_match:
                        print("car detected")
                        car.detected = True
                        car.number_of_detections += 1
                        car.status.append(car.detected)
                        car.status = car.status[-KEEP_LAST_STATUS:]
                    else:  # car is not detected
                        print("not matched")
                        car.detected = False
                        car.status.append(car.detected)
                        car.status = car.status[-KEEP_LAST_STATUS:]
                        if not car.is_new:  # if car is not new draw rectangle
                            # if intersect exist than add it to the list
                            if x_intersect_ratio > 0.3 and y_intersect_ratio > 0.3:
                                car.xpixels = nonzerox
                                car.ypixels = nonzeroy

                                car.recent_xfitted.append(car.xpixels)
                                car.recent_xfitted = car.recent_xfitted[-NUM_ITERATIONS_TO_KEEP:]

                                car.recent_yfitted.append(car.ypixels)
                                car.recent_yfitted = car.recent_yfitted[-NUM_ITERATIONS_TO_KEEP:]

                                minx_array = [np.min(x) for x in car.recent_xfitted]
                                miny_array = [np.min(y) for y in car.recent_yfitted]
                                maxw_array = [np.max(x) for x in car.recent_xfitted]
                                maxh_array = [np.max(y) for y in car.recent_yfitted]

                                car.bestx = int(np.mean(minx_array))
                                car.besty = int(np.mean(miny_array))
                                car.bestw = int(np.mean(maxw_array))
                                car.besth = int(np.mean(maxh_array))

                            top_left = (car.bestx, car.besty)
                            bottom_right = (car.bestw, car.besth)
                            bbox = (top_left, bottom_right)
                            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
                            cv2.putText(img, "detected : {}".format(car.detected), (10, 60),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
                            cv2.putText(img, "x_ratio : {} / y_ratio : {}".format(x_intersect_ratio, y_intersect_ratio),
                                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

                            cv2.putText(img, "num detected cars : {}".format(labels[1]),
                                        (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
                            continue
                        else:
                            print("Car yeni detect edilen obje o nedenle çizilemedi")
                            continue  # car is new do nothing

                car.status.append(car.detected)
                car.status = car.status[-KEEP_LAST_STATUS:]

                car.xpixels = nonzerox
                car.ypixels = nonzeroy

                car.recent_xfitted.append(car.xpixels)
                car.recent_xfitted = car.recent_xfitted[-NUM_ITERATIONS_TO_KEEP:]

                car.recent_yfitted.append(car.ypixels)
                car.recent_yfitted = car.recent_yfitted[-NUM_ITERATIONS_TO_KEEP:]

                minx_array = [np.min(x) for x in car.recent_xfitted]
                miny_array = [np.min(y) for y in car.recent_yfitted]
                maxw_array = [np.max(x) for x in car.recent_xfitted]
                maxh_array = [np.max(y) for y in car.recent_yfitted]

                car.bestx = int(np.mean(minx_array))
                car.besty = int(np.mean(miny_array))
                car.bestw = int(np.mean(maxw_array))
                car.besth = int(np.mean(maxh_array))

                if len(car.status) >= 10:
                    top_left = (car.bestx, car.besty)
                    bottom_right = (car.bestw, car.besth)
                    bbox = (top_left, bottom_right)
                    cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
                    cv2.putText(img, "detected : {}".format(car.detected), (10, 60),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
                    cv2.putText(img, "x_ratio : {} / y_ratio : {}".format(x_intersect_ratio, y_intersect_ratio),
                                (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

                    cv2.putText(img, "num detected cars : {}".format(labels[1]),
                                (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

                else:
                    print("Detect edilen status sayısı 10 'dan  küçük Dikdörtgen çizilmedi")

    else:
        print("""!!!!!!!!!!!!!!!!!!!!!!!!!!!! No cars detected in labels !!!!!!!!!!!!!!!!!!!!!!!!1""")
        if len(carslist) > 0:
            for car in carslist:
                top_left = (car.bestx, car.besty)
                bottom_right = (car.bestw, car.besth)
                bbox = (top_left, bottom_right)
                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
                cv2.putText(img, "detected : {}".format(car.detected), (10, 60),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
                cv2.putText(img, "x_ratio : {} / y_ratio : {}".format(x_intersect_ratio, y_intersect_ratio),
                            (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

                cv2.putText(img, "num detected cars : {}".format(labels[1]),
                            (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

    # # remove cars if detected all False in status array
    # for car in carslist:
    #     if len(set(car.status)) == 1 and len(car.status) >= 15:
    #         if not car.status[0]:
    #             print(car.status)
    #             carslist.remove(car)
    #             print("car removed because last 10 detection is false")

    # remove cars if has more than detected
    if len(carslist) > labels[1]:
        difference = len(carslist)-labels[1]
        cars_to_be_deleted = carslist[-difference:]
        for car in cars_to_be_deleted:
            carslist.remove(car)
            print("car removed because there are more cars than detected.")

    for car in carslist:  # bunu silersen diğerleri çalışmıyor...
        if len(car.status) >= 10:
            car.is_new = False
    # Return the image
    return img


# class Vehicle:
#     def __init__(self):
#         self.is_new = True
#         self.is_new = False
#         self.car_number = 0
#         self.detected = deque(maxlen=10)

#
# def draw_labeled_bboxes(img, labels, min_rect_size=2500, min_aspect_ratio=0.6, max_aspect_ratio=2):
#     # Iterate through all detected cars
#     for car_number in range(1, labels[1] + 1):
#         car = carslist[car_number - 1]
#         # Find pixels with each car_number label value
#         nonzero = (labels[0] == car_number).nonzero()
#         # Identify x and y values of those pixels
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         # Define a bounding box based on min/max x and y
#         bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
#         # Draw the box on the image
#         bbox_rect_size = (bbox[1][0]-bbox[0][0]) * (bbox[1][1]-bbox[0][1])
#         bbox_aspect_ratio = (bbox[1][0]-bbox[0][0]) / (bbox[1][1]-bbox[0][1])
#
#         if (bbox_rect_size > min_rect_size) and (bbox_aspect_ratio > min_aspect_ratio) and (bbox_aspect_ratio < max_aspect_ratio):
#             car.detected.append(True)
#             if len(set(car.detected)) == 1:  # all are the same
#                 print("all are the same")
#                 if car.detected[0]:
#                     print("all are True")
#                     cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
#         else:
#             if bbox_rect_size < min_rect_size:
#                 print("bbox_rect_size küçük bu box elendi...")
#                 print("bbox_rect_size :", bbox_rect_size)
#                 car.detected.append(False)
#             if bbox_aspect_ratio < min_aspect_ratio:
#                 print("bbox_aspect_ratio küçük bu box elendi...")
#                 print("aspect_ratio :", bbox_aspect_ratio)
#                 car.detected.append(False)
#             if bbox_aspect_ratio > max_aspect_ratio:
#                 print("bbox_aspect_ratio büyük bu box elendi...")
#                 print("aspect_ratio :", bbox_aspect_ratio)
#                 car.detected.append(False)
#         print("car_number", car_number, "car.number_of_detections", car.detected)
#
#
#     return img


def process_image(img):

    # print(np.min(img), np.max(img))  # make sure that image is in range between (0, 255)
    for scale, ystart, ystop in scales:
        out_img, heat_map = hf.find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                         pix_per_cell, cell_per_block, cells_per_step, spatial_size, hist_bins)
        # print(np.min(heat_map), np.max(heat_map))
        heatmaps_collection.append(heat_map)
    sum_of_heatmaps = sum(heatmaps_collection[-40:])  # sum of last X
    sum_of_heatmaps = hf.apply_threshold(sum_of_heatmaps, threshold=90)
    # average_heat_map = sum_of_heatmaps / len(heatmaps_collection)
    # print("average_heat_map :", average_heat_map.nonzero())
    average_heat_map = np.clip(sum_of_heatmaps, 0, 255)
    # print(np.min(average_heat_map), np.max(average_heat_map))
    labels = label(average_heat_map)
    print("np.max(labels)", np.max(labels[0]), labels[1])
    number_of_detected_cars = labels[1]
    # print("number_of_detected_cars", number_of_detected_cars)

    number_of_missing_vehicles = number_of_detected_cars - len(carslist)

    if number_of_missing_vehicles >= 0:
        for i in range(1, number_of_missing_vehicles+1):
            car = Vehicle()
            car.car_number = i
            carslist.append(car)

    draw_img = draw_labeled_bboxes(np.copy(img), labels, carslist)

    return draw_img


filename = "project_video.mp4"
# clip1 = VideoFileClip(filename).subclip(40, 45)
clip1 = VideoFileClip(filename).subclip(25, 35)
# clip1 = VideoFileClip(filename)
clip2 = clip1.fl_image(process_image)
clip2.write_videofile("output_video/v4_{}".format(filename), audio=False)


