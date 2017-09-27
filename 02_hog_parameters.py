import helper_functions as hf
import numpy as np
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


# Divide up into cars and notcars
cars_train = hf.vehicle_images()
notcars_train = hf.non_vehicle_images()

# Select 1000 random cars and not cars
sample_size = 1000

# generate random indexes for 1000 samples
random_indexes = list(np.random.randint(0, len(cars_train), sample_size))

test_cars = []
test_notcars = []
for index in random_indexes:
    test_cars.append(cars_train[index])
    test_notcars.append(notcars_train[index])

cars = test_cars
notcars = test_notcars

# Play with these parameters
colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

t = time.time()
try:
    dist_pickle = pickle.load(open("feat_pickle.p", "rb"))
    car_features = dist_pickle["car_features"]
    notcar_features = dist_pickle["notcar_features"]
except:
    car_features = hf.extract_features(cars, color_space=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)
    notcar_features = hf.extract_features(notcars, color_space=colorspace, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          hog_channel=hog_channel)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', colorspace, 'colorspace,', orient, 'orientations,', pix_per_cell,
      'pixels per cell,', cell_per_block, 'cells per block and', hog_channel, 'as hog channel parameter.')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


# Save training parameters
dist_pickle = dict()
dist_pickle["car_features"] = car_features
dist_pickle["notcar_features"] = notcar_features
pickle.dump(dist_pickle, open("feat_pickle.p", "wb"))
