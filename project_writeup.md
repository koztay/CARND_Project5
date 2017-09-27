##Project Writeup - Vehicle Detection Project




The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[image1]: ./output_images/00_car_notcar.png
[image2]: ./output_images/01_HOG_visualization.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called `01_hog.py`. I also created another file named `helper_functions.py` and put all the common functions for the project into this file. 

I started by reading in all the `vehicle` and `non-vehicle` images. For this purpose I used `data_look()` function from the lessons `(lines between 10-27 in 01_hog.py)` and `vehicle_images()` and `non_vehicle_images()` from `helper_functions.py file (lines between 18-35)`. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed again random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. `01_hog_features.py lines between 55 and 110`

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

For defining the best HOG parameters I have created `02_hog_parameters.py` file in order to try various combinations of parameters. In order to reduce the time of the trials first I have selected the small amount of data to train `(lines between 14-32)`, second I pickled the features vector and read for trials `(lines between 85-90 for pickle and lines between 40-51 for reading from pickle file.)`

First I tried RGB colorspace with following parameters and after that I played with many of them and the results were like the following:

```
Using: RGB colorspace, 8 orientations, 8 pixels per cell, 
2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
9.82 Seconds to train SVC...
Test Accuracy of SVC =  0.9875
```

```
Using: RGB colorspace, 8 orientations, 16 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
9.64 Seconds to train SVC...
Test Accuracy of SVC =  0.9875
```

```
Using: RGB colorspace, 8 orientations, 4 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
8.52 Seconds to train SVC...
Test Accuracy of SVC =  0.98
```

```
Using: RGB colorspace, 8 orientations, 16 pixels per cell, 
4 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
2.9 Seconds to train SVC...
Test Accuracy of SVC =  0.9875
```

As seen from the results above, playing with pix_per_cell and cell_per_blok almost has not or has very small effect on the accuracy. So, I decided to focus on playing with others.

```
Using: HSV colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
5.49 Seconds to train SVC...
Test Accuracy of SVC =  0.9925
```

```
Using: LUV colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
8.56 Seconds to train SVC...
Test Accuracy of SVC =  0.985
```  

```
Using: HLS colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
8.73 Seconds to train SVC...
Test Accuracy of SVC =  0.99
```

```
Using: YUV colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
5.16 Seconds to train SVC...
Test Accuracy of SVC =  0.9875
```

```
Using: YCrCb colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and 0 as hog channel parameter.
Feature vector length: 8460
8.4 Seconds to train SVC...
Test Accuracy of SVC =  0.9875
```

```
Using: YCrCb colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and 1 as hog channel parameter.
Feature vector length: 8460
8.81 Seconds to train SVC...
Test Accuracy of SVC =  0.985
```

```
Using: YCrCb colorspace, 12 orientations, 8 pixels per cell, 2 cells per block and ALL as hog channel parameter.
Feature vector length: 8460
8.8 Seconds to train SVC...
Test Accuracy of SVC =  0.995
```

After several trials, finally I came up with the best accuracy results
with "YCrCb" colorspace, "12" orientations, "8" pixels per cell, "2" cells per block and "ALL" as hog channel parameter. The accuracy was "0.995" with these paramaters.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the file name `03_training_SVM.py ` . I was saved the features data as a pickle pickle file   the features Finally I saved the classifier data (svc, X_scaler etc.) as a pickle file in order to use it later on predictions `(lines between 60 and 70)`. 
After training classifier for whole data using the parameters defined in the previous step, the accuracy was 0.9913

At this step I created the feature vectors from scratch because the feature vectors saved as a pickle file does not contain the whole dataset. So, the statistics for the training session were as follows:

```
97.09 Seconds to extract HOG features...
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 10224
12.69 Seconds to train SVC...
Test Accuracy of SVC =  0.9913
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
