import helper_functions as hf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler


# read / plot random vehicle and non-vehicle images use data_look func from lectures
def data_look(car_list, notcar_list):

    data_dict = dict()
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    car_ind = np.random.randint(0, len(cars))
    test_img = mpimg.imread(cars[car_ind])
    img_shape = test_img.shape
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = img_shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = test_img.dtype
    # Return data_dict
    return data_dict


cars = hf.vehicle_images()
notcars = hf.non_vehicle_images()
data_info = data_look(cars, notcars)

print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

images = [car_image, notcar_image]
titles = ["Random Car Image", "Random Not Car Image"]
# Plot the examples
fig = plt.figure(figsize=(10, 5))
hf.visualize(fig, 1, 2, images, titles)
plt.show()
fig.savefig("output_images/00_car_notcar.png")


car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
ycrcb_car = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
car_channel_1 = ycrcb_car[:, :, 0]
car_channel_2 = ycrcb_car[:, :, 1]
car_channel_3 = ycrcb_car[:, :, 2]

ycrcb_not_car = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YCrCb)
not_car_channel_1 = ycrcb_not_car[:, :, 0]
not_car_channel_2 = ycrcb_not_car[:, :, 1]
not_car_hannel_3 = ycrcb_not_car[:, :, 2]

# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2

# Call our function with vis=True to see an image output
features_car, hog_image_car = hf.get_hog_features(car_channel_1, orient,
                                                  pix_per_cell, cell_per_block,
                                                  vis=True, feature_vec=True)

features_not_car, hog_image_not_car = hf.get_hog_features(not_car_channel_1, orient,
                                                          pix_per_cell, cell_per_block,
                                                          vis=True, feature_vec=True)
print(hog_image_car.shape)
print(features_car.shape)

img0 = car_image
img1 = hog_image_car
img2 = car_channel_1
img3 = car_channel_2
img4 = car_channel_3
img5 = notcar_image
img6 = hog_image_not_car
img7 = not_car_channel_1
img8 = not_car_channel_2
img9 = not_car_hannel_3


images = [img0, img1, img2, img3, img4, img5, img6, img7, img8, img9]
titles = ["Image", "Hog Image", "Y Channel", "Cr Channel", "Cb Channel",
          "Not Car Image", "Hog Image", "Y Channel", "Cr Channel", "Cb Channel"]

fig = plt.figure(figsize=(10, 4))
hf.visualize(fig, 2, 5, images, titles)
plt.show()

fig.savefig("output_images/01_hog_visualization.png")

