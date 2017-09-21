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

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.show()
fig.savefig("output_images/00_car_notcar.png")


# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])

# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hls = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
channel_1 = hls[:, :, 0]
channel_2 = hls[:, :, 1]
channel_3 = hls[:, :, 2]

# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2

# Call our function with vis=True to see an image output
features, hog_image = hf.get_hog_features(channel_3, orient,
                                          pix_per_cell, cell_per_block,
                                          vis=True, feature_vec=True)


print(hog_image.shape)
print(features.shape)

# Plot the examples
fig = plt.figure()
plt.subplot(331)
plt.imshow(channel_1, cmap='gray')
plt.title('Example Car Image')
plt.subplot(332)
plt.imshow(channel_2, cmap='gray')
plt.title('HOG Visualization')
plt.subplot(333)
plt.imshow(channel_3, cmap='gray')
plt.title('Channel2 Visualization')


plt.show()
fig.savefig("output_images/01_hog_visualization.png")

