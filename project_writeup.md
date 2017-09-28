##Project Writeup - Vehicle Detection Project




The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[video1]: ./output_video/project_video.mp4
[image1]: ./output_images/00_car_notcar.png
[image2]: ./output_images/01_HOG_visualization.png
[image3]: ./output_images/04_sliding_window_search_0_50.png
[image4]: ./output_images/04_sliding_window_search_0_75.png
[image5]: ./output_images/04_sliding_window_search_0_50_64by64.png
[image6]: ./output_images/04_sliding_window_search_0_75_64by64.png
[image7]: ./output_images/04_sliding_window_search_0_75_96by96.png
[image8]: ./output_images/05_images_pipeline_128by128.png
[image9]: ./output_images/05_images_pipeline_64by64.png
[image10]: ./output_images/05_images_pipeline_combined_scales.png

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

I created a file named `04_sliding_window_search.py` in order to define the parameters for what scales to search and how much overlapping windows. Here below the results of several parameter trials:

**0.50 Overlap / 128x128 px window size :** 
![alt text][image3]

**0.75 Overlap / 128x128 px window size :** 
![alt text][image4]

**0.50 Overlap / 64x64 px window size :** 
![alt text][image5]

**0.75 Overlap / 64x64 px window size :** 
![alt text][image6]

The best results were achieved by 0.75 overlap with 128x128 pixels window size.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Not: From now on I used the `find_cars()` function from the lessons with my file `05_images_pipeline.py`. 
Ultimately I searched on two scales `(scale=1 which means 64x64 and scale=2 which means 128x128)` using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are the results of example images with each scales:

**0.75 Overlap / 128x128 px window size :** 
![alt text][image8]

**0.75 Overlap / 64x64 px window size :** 
![alt text][image9]

In order to optimize performance of my classifier I collected heatmaps for each scale and applied threshold for the collected heatmaps. Here the result of combined thresholds for each scales:

**0.75 Overlap / Combined Scales :**
![alt text][image10]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For video pipeline I have created another file named `06_video_pipeline.py`. 
And first I decided to work on false positives. In order to remove false positivies I have summed the heatmaps over several frames and after that applied threshold and averaged them. `lines between 225-232`. After that I applied `scipy.ndimage.measurements.label()` function in order to identify individual blobs in the heatmap. 
(I already plotted the heatmaps at the previous section so, I will not plot the heaptmaps again.)

By doing that most of the false positives were removed but I could not succeed to get rid of all the false positives. Because if I increase the threshold then I get smaller sized bounding boxes around cars and in some areas I got several windows on the same cars especially on big ones. In order to get rid of this I will add the following lines to my `find_cars()` function `between the lines 424-430`:

```python
if len(img_boxes) > 0:
    for box in img_boxes:
        top_left, bottom_right = box
        if np.max(heatmap[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]) > 1:
            heatmap[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] += 2

```

The above lines adds 2 more heatmap if there is more than 1 window detected so, it makes easily to remove the False positives remaining. After adding the above lines, I achived a video wihout any False positives.

But there was one last problem and it was wobbly boxes problem. In order to solve this problem I created a Vehicle class `between the lines of 29-51` in order to keep the status of cars and detections. But I also needed to modify `draw_labeled_bboxes()` function an when it becomes a huge function I removed it form my helper_functions.py and put it into the `06_video_pipeline.py  (lines between 53 and 200)`.

Finally I achieved a smooth moving bounding boxes and no any False positive detections.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

