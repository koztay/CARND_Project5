import helper_functions as hf
import numpy as np
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler


# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


# Divide up into cars and notcars
cars = hf.vehicle_images()
notcars = hf.non_vehicle_images()

# TODO: Tweak these parameters and see how the results change.
colorspace = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

t = time.time()

try:
    feat_pickle = pickle.load(open("feat_pickle.p", "rb"))
    car_features = feat_pickle['car_features']
    notcar_features = feat_pickle['notcar_features']
    print("features pickle exists and used existing values")
except:
    car_features = hf.extract_features(cars, color_space=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)
    notcar_features = hf.extract_features(notcars, color_space=colorspace, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          hog_channel=hog_channel)
    feat_pickle = dict()
    feat_pickle['car_features'] = car_features
    feat_pickle['notcar_features'] = notcar_features
    pickle.dump(feat_pickle, open("feat_pickle.p", "wb"))


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

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
# svc = LinearSVC()
# parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 3, 5, 7, 10]}
svc = SVC()
# svc = grid_search.GridSearchCV(svr, parameters)
# clf.fit(X_train, y_train)

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)

# print("best params", svc.best_params_)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
print('My SVC predicts: ', svc.predict(X_test))
print('For these', len(X_test), 'labels: ', y_test)
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', len(X_test), 'labels with SVC')

# Save training parameters
dist_pickle = dict()
dist_pickle["svc"] = svc
# dist_pickle["best_params"] = svc.best_params_
dist_pickle["scaler"] = X_scaler
dist_pickle["color_space"] = colorspace
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["hog_channel"] = hog_channel

pickle.dump(dist_pickle, open("svc_pickle.p", "wb"))


"""
98.39 Seconds to extract HOG features...
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 10224
15.84 Seconds to train SVC...
Test Accuracy of SVC =  0.9901
My SVC predicts:  [ 0.  0.  1.  0.  0.  0.  0.  1.  1.  0.  1.  1.  0.  1.  1.  0.  1.  1.
  1.  1.  0.  1.  0.  1.  0.  0.  1.  1.  0.  0.  0.  1.  0.  1.  1.  0.
  0.  0.  1.  1.  1.  1.  1.  0.  0.  0.  1.  0.  0.  1.  0.  1.  1.  1.
  1.  0.  1.  0.  0.  1.  1.  0.  0.  1.  0.  1.  1.  1.  0.  0.  0.  0.
  0.  1.  0.  0.  1.  1.  1.  0.  1.  0.  1.  1.  1.  0.  0.  0.  0.  0.
  0.  0.  1.  0.  0.  1.  0.  0.  0.  0.]
For these 100 labels:  [ 0.  0.  1.  0.  0.  0.  0.  1.  1.  0.  1.  1.  0.  1.  1.  0.  1.  1.
  1.  1.  0.  1.  0.  1.  0.  0.  1.  1.  0.  0.  0.  1.  0.  1.  1.  0.
  0.  1.  1.  1.  1.  1.  1.  0.  0.  0.  1.  0.  0.  1.  0.  1.  1.  1.
  1.  0.  1.  0.  0.  1.  1.  0.  0.  1.  0.  1.  1.  1.  0.  0.  0.  0.
  0.  1.  0.  0.  1.  1.  1.  0.  1.  0.  1.  1.  1.  0.  0.  0.  0.  0.
  0.  0.  1.  0.  0.  1.  0.  0.  0.  0.]
0.01414 Seconds to predict 100 labels with SVC

Process finished with exit code 0

"""