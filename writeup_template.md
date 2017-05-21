**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in the second section in the cell number 6. In cell number 7 there is an example of the use and the ouput image, also attached here:  

![hog feature](./examples/hog_feature.png)

In the final solution I will use `HLS` as color space, and use all the channels to get the HOG features.

The data for the `vehicle` and `non_vehicle` images have been downloaded from the lesson data. In the utils section there is a function who reads both type of images and return two arrays with the two groups. 

The method `get_hot_features` can have two possible outputs, depending on the input argument `vis`, that basically indicates if we can also get an output image for the HOG visualization. We will pass `true` just for the test and keep the value to `false` in other case.
During the realization of the project I used different values for `orientations`, `pixels_per_cell` and `cells_per_block`, and finally choose `12`, `4` and `2` respectively. 

####2. Explain how you settled on your final choice of HOG parameters.

The only way I had to check how the HOG parameters affect my choice was testing in my classifier. 
As mentioned before, I read the image data and used to get the HOG features for `vehicles` and `non_vehicles` images.
After several tweaking and testing different options, I got the best results with the following values:

`color space` -> HLS
`orientations` -> 12
`pixels_per_cell` -> 4
`cells_per_block` -> 2
`hog_channel` -> ALL

Increasing the orientation, using all the channels from the image to get the features, together with smaller blocks and less pixels for cell, made the process of getting the vector features considerably slow in my laptop, but got the best prediction with this values

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

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

