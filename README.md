
# CarND Term 1: Advanced Lane Line Detection

In this notebook, I walk through a method for detecting lane lines from a video stream using common computer vision techniques. 




```python
import numpy as np
import cv2
import glob

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```

## Camera Calibration

The first step of our pipeline involves calculating the distortion of our camera images. To calculate the camera's distortion coefficients, we use an image with known measurements (in this case, a checkerboard pattern) and measure the difference between the two. 

First we define our "object points", which are the corners of the chessboard. `objp` is our array of coordinates that we'll measure our camera images against. In this case, it's a 6 x 9 chessboard.


```python
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
```

Next, we load in the calibration images (photos of the chessboard taken by our camera) and use cv2's `findChessBoardCorners` function to detect the location of the corners in our image. 

Using the camera image corners and the "real-world" coordinates, we use cv2's `calibrateCamera` function to compute the distortion coefficient. It returns the root mean squared error of the calibration, the camera matrix, the distortion coefficients, rotation vectors, and translation vectors.


```python
images = glob.glob('camera_cal/calibration*.jpg')

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    # If found, add object points, image points
    if found == True:
        objpoints.append(objp)
        imgpoints.append(corners)

## Calculate distortion
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

```

This is what the original and undistorted images look like. 


```python
orig_img = cv2.imread(images[0])
## Undistort using coefficients calculated previously
new_img = cv2.undistort(orig_img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(orig_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(new_img)
ax2.set_title('Undistorted Image', fontsize=30)
```




    <matplotlib.text.Text at 0x10eaaed30>




![png](output_7_1.png)


## Single Image Pipeline

### Distortion correction
We apply this same distortion correction to our test images, as per the xample below. 



```python
test_img = cv2.imread('test_images/straight_lines1.jpg')
test_img_undist = cv2.undistort(test_img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(test_img_undist)
ax2.set_title('Undistorted Image', fontsize=30)
```




    <matplotlib.text.Text at 0x11058c278>




![png](output_9_1.png)


### Gradient calculation

To transform our undistorted image into a binary image, I used the Sobel operator, which takes the gradient of each pixel in either the x or y direction. It appears that the Sobel operator in the x-axis did a better job of isolating the lane lines from other shapes in the image. 


```python
def abs_sobel_thresh(img, orient, thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=="x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient=="y":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        raise Exception("Orient must be x or y.")
        
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output
    

test_img_binx = abs_sobel_thresh(test_img_undist, "x", thresh_min=25, thresh_max=100)
test_img_biny = abs_sobel_thresh(test_img_undist, "y", thresh_min=25, thresh_max=100)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.imshow(test_img_undist)
ax1.set_title('Image', fontsize=20)
ax2.imshow(test_img_binx, cmap="gray")
ax2.set_title('Binary Image (Sobel X)', fontsize=20)
ax3.imshow(test_img_biny, cmap="gray")
ax3.set_title('Binary Image (Sobel Y)', fontsize=20)
```




    <matplotlib.text.Text at 0x117a48dd8>




![png](output_11_1.png)


### Perspective Transform

In the next step of the process, we apply a perspective transform in order to get a birds-eye view of the road ahead. I used hard-coded coordinates to construct the transformation. 


```python
# Define the coordinates of a region in our source image
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])

# Define the corresponding coordinates of our target image
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])


img = cv2.imread('test_images/test6.jpg')
img = cv2.undistort(img, mtx, dist, None, mtx)

transform_M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, transform_M, img_size, flags=cv2.INTER_LINEAR)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(warped)
ax2.set_title('Image after perspective transform', fontsize=20)
```




    <matplotlib.text.Text at 0x112320550>




![png](output_13_1.png)


Combining the steps demonstrated above gives us the image processing pipeline necessary for drawing the lane regions of the image. 

1. Camera undistortion
2. Gradient thresholding with Sobel
3. Perspective transform


```python
img = cv2.imread('test_images/test6.jpg')

def transform_image(img, camera_mtx, camera_dist, perspective_mtx, s_thresh=(0, 255)):
    ## Get image size in length by height
    img_size = (img.shape[1], img.shape[0])
    
    ## Camera distortion correction
    img = cv2.undistort(img, camera_mtx, camera_dist, None, camera_mtx)
    
    ## Sobel operator
    img = abs_sobel_thresh(img, "x", s_thresh[0], s_thresh[1])
    
    ## Perspective transform
    img = cv2.warpPerspective(img, perspective_mtx, img_size, flags=cv2.INTER_LINEAR)
    
    return img

warped = transform_image(img, mtx, dist, transform_M, (25, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(warped, cmap="gray")
ax2.set_title('Final image', fontsize=20)
```




    <matplotlib.text.Text at 0x11260ca20>




![png](output_15_1.png)


### Lane Detection

Next we detect the presence of lane lines in our transformed, thresholded image. To do this we first find the base of our line by summing up the non-zero pixels height-wise from the bottom half of the image and finding the peaks. 


```python
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)

def get_line_base(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    return leftx_base, rightx_base    
        
leftx_base, rightx_base = get_line_base(warped)

print ("Left lane line starts at: ", leftx_base)
print ("Right lane line starts at: ", rightx_base)
```

    Left lane line starts at:  399
    Right lane line starts at:  1045



![png](output_17_1.png)


We use the bases of the lines to narrow our search over the rest of the image. We first divide the image into horizontal sections. For each lane line, we iterate through each section starting from the bottom, using the previously detected lane midpoint to draw a window over which we select the non-zero pixels for the lane line (if there aren't enough non-zero pixels in the window in question, it draws the next window around the previously determined x coordinate). 

This is all implemented in the `sliding_window_search` function. 



```python
def sliding_window_search(nonzerox, nonzeroy, x_start, window_height, nwindows, margin=50, minpix=50):
    
    x_current=x_start
    
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
    
    lane_inds = np.concatenate(lane_inds)
    return lane_inds

## Get x and y coordinates of non-zero pixels
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

## Get height of the windows used for the sliding window search
nwindows = 9
window_height = np.int(warped.shape[0]//nwindows)

left_lane_inds = sliding_window_search(nonzerox, nonzeroy, leftx_base, window_height, nwindows, margin=50, minpix=50)
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 

right_lane_inds = sliding_window_search(nonzerox, nonzeroy, rightx_base, window_height, nwindows, margin=50, minpix=50)
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
```

Next, we use numpy's `polyfit` function to fit a 2nd-degree polynomial curve using our detected left lane pixels and right lane pixels.


```python

def calculate_current_fit(x_pixels, y_pixels, height):
    fit =  np.polyfit(y_pixels, x_pixels, 2)
    yfitted = np.linspace(0, height-1, num=height)
    xfitted = fit[0]*yfitted**2 + fit[1]*yfitted + fit[2] 
    return xfitted, yfitted

leftx, lefty = calculate_current_fit(leftx, lefty, warped.shape[0])
rightx, righty = calculate_current_fit(rightx, righty, warped.shape[0])

plt.imshow(warped, cmap="gray")
plt.plot(leftx, lefty, color='red')
plt.plot(rightx, righty, color='red')
plt.xlim(0, warped.shape[1])
plt.ylim(warped.shape[0], 0)
```




    (720, 0)




![png](output_21_1.png)


To calculate curvature, we use the following equation: 

<img src="examples/Curvature.png">

where A and B are the coefficients on the second- and first-degree terms of our polynomial fit. To calculate the curvature in real-life units, I convert from pixel space to meters in the y and x dimensions with the parameters `ym_per_pix` (meters per y pixel) and `xm_per_pix` (meters per x pixel). 

To calculate the relative position of the car, we convert the x-axis from pixels to meters, get the position of the center of the lane by averaging the base of the right and left lines, and subtract it from the midpoint of the camera image (which we assume to be positioned at the center of the car). 




```python
def calculate_radius(x_val, y_val, 
                     ym_per_pix = 30./720, 
                     xm_per_pix = 3.7/700):
    y_eval = np.max(y_val)
    fit_cr = np.polyfit(y_val*ym_per_pix, x_val*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    
    return curverad

```

Finally, we shade in the region between our detected lane lines and draw the detected lane region back onto the original image in the `draw_lane` function.

`process_frame` contains the full pipeline for transforming and detecting the lanes in a given image. 


```python
def draw_lane(warped, lline_x, lline_y, rline_x, rline_y, Minv, img_shape):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lline_x, lline_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rline_x, rline_y])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[0], img_shape[1])) 
    return newwarp


def process_frame(img, mtx, dist, transform_M, s_thresh, nwindows=9):
    
    warped = transform_image(img, mtx, dist, transform_M, s_thresh)
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    window_height = np.int(warped.shape[0]//nwindows)

    left_lane_inds = sliding_window_search(nonzerox, nonzeroy, leftx_base, window_height, nwindows, margin=50, minpix=50)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 

    right_lane_inds = sliding_window_search(nonzerox, nonzeroy, rightx_base, window_height, nwindows, margin=50, minpix=50)
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    leftx, lefty = calculate_current_fit(leftx, lefty, warped.shape[0])
    rightx, righty = calculate_current_fit(rightx, righty, warped.shape[0])
    
    lane = draw_lane(warped, leftx, lefty, rightx, righty, np.linalg.inv(transform_M), (img.shape[1], img.shape[0]))
    
    final_img = cv2.addWeighted(img, 1, lane, 0.3, 0)
    
    lrad = calculate_radius(leftx, lefty)
    rrad = calculate_radius(rightx, righty)
    
    ## Get road curvature and car position
    curve =  (lrad + rrad)/2
    lane_center_pos = (leftx[-1] + rightx[-1])/2 * 3.7/700
    car_center_pos = (img.shape[1]/2)*3.7/700
    car_relative_pos = (car_center_pos - lane_center_pos) * 100
    info_text = "Curvature radius: {0} m; Car position: {1} cm".format(("%3.f" % curve), ("%2.f" % car_relative_pos))
    
    final_img = cv2.putText(final_img, info_text, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    
    return final_img

final = process_frame(img, mtx, dist, transform_M, (25, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(final, cmap="gray")
ax2.set_title('Final image', fontsize=20)
```




    <matplotlib.text.Text at 0x1179ef470>




![png](output_25_1.png)


### Video Pipeline

When applying our image processing pipeline to consecutive frames from a video stream, I added several enhacements. 

- Improved lane pixel search: Instead of detecting the base of the line and using the sliding window search technique for each frame, we use the previously detected line (if it exists) to define a window around which to detect lane pixels in the current frame. 



```python
def last_line_search(fit, nonzerox, nonzeroy, margin=100):

    lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) 
                 & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin))) 

    return lane_inds

```

We define a class called `Line` that contains the other changes: 

- Sanity check: Check results from each frame and if they fall outside certain acceptable thresholds, carry forward the previously detected frame. This is the `calculate_current_fit` method.
- Smoothing: We average the fitted lane lines from the last N frames to avoid jumpy lane regions in our output video. This is the `get_smoothed_fit` method.


```python
class Line(object):
    def __init__(self, ylen, linetype, memory):
        ## Right or left
        self.linetype = linetype
        # was the line detected in the last iteration?
        self.detected = False 
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # x values of current x fit 
        self.current_xfitted = None
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        ## Y coordinates
        self.ploty = np.linspace(0, ylen-1, num=ylen)
        
        ## Number of previous fits to apply smoothing over
        self.memory = memory
        
    def detect_line(self, img, nwindows):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        window_height = np.int(warped.shape[0]//nwindows)
        
        if self.detected == False:
            if self.linetype=="left":
                line_base, _ = get_line_base(img)
            else:
                _, line_base = get_line_base(img)
            lane_inds = sliding_window_search(nonzerox, nonzeroy, line_base, window_height, nwindows, margin=50, minpix=50)
        else: 
            lane_inds = last_line_search(self.best_fit, nonzerox, nonzeroy)
        
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds] 
        
        if self.allx.size == 0 or self.ally.size == 0:
            self.detected=False
        else:
            self.detected=True
            self.calculate_current_fit()
        
        self.get_smoothed_fit()
        return
        
    def calculate_current_fit(self):
        fit =  np.polyfit(self.ally, self.allx, 2)
        
        ## Check if curvature radius is within 500 meters of the previous frame. 
        rad = calculate_radius(fit[0]*self.ploty**2 + fit[1]*self.ploty + fit[2], self.ploty)
        if self.radius_of_curvature is not None and abs(rad - self.radius_of_curvature) > 50:
            self.detected=False
            self.radius_of_curvature = rad
            return
        
        ## Check if position of the line is within 2 meters of the previous frame
        
        line_base_pos = (fit[0]*self.ploty[-1]**2 + fit[1]*self.ploty[-1] + fit[2])* 3.7/700
        if self.line_base_pos is not None and abs(line_base_pos - self.line_base_pos) > .2 :
            self.detected=False
            return
        
        self.current_fit = fit
        self.current_xfitted = fit[0]*self.ploty**2 + fit[1]*self.ploty + fit[2] 
        return
    
    def get_smoothed_fit(self):
        if self.detected:
            self.recent_xfitted.append(self.current_xfitted)
        else:
            self.recent_xfitted = [] 
            self.recent_xfitted += self.recent_xfitted[-1:]
        self.recent_xfitted = self.recent_xfitted[-self.memory:]
        if len(self.recent_xfitted)>0:
            self.bestx = np.mean(self.recent_xfitted, axis=0)
            self.best_fit = np.polyfit(self.ploty, self.bestx, 2)
            self.line_base_pos = self.bestx[-1] * 3.7/700
            self.radius_of_curvature = calculate_radius(self.best_fit[0]*self.ploty**2 + self.best_fit[1]*self.ploty + self.best_fit[2], self.ploty)
        return
```

Finally we wrap our entire pipeline in a FrameProcessor object that contains two `Line` objects for the right and left lane lines, and performs the image transformation, lane detection, and sanity checking/smoothing discussed above.  


```python
class FrameProcessor(object):
    
    def __init__(self, mtx, dist, transform_M, img_shape, nwindows=9, memory=10):
        self.lline = Line(img_shape[0], "left", memory)
        self.rline = Line(img_shape[0], "right", memory)
        self.mtx = mtx
        self.dist = dist
        self.M = transform_M
        self.Minv = np.linalg.inv(transform_M)
        self.img_shape = img_shape
        self.nwindows = nwindows
        
    def get_lane_info(self):
        curve =  (self.lline.radius_of_curvature + self.rline.radius_of_curvature)/2
        lane_center_pos = (self.lline.line_base_pos + self.rline.line_base_pos)/2
        car_center_pos = (self.img_shape[0]/2)*3.7/700
        car_relative_pos = (car_center_pos - lane_center_pos)*100
        text = "Curvature radius: {0} m; Car position: {1} cm".format(("%2.f" % curve), ("%2.f" % car_relative_pos))
        return text

    def process_frame(self, img, s_thresh=(25, 100)):

        warped = transform_image(img, self.mtx, self.dist, self.M, s_thresh)
        
        self.lline.detect_line(warped, self.nwindows)
        self.rline.detect_line(warped, self.nwindows)

        lane = draw_lane(warped, self.lline.bestx, self.lline.ploty, self.rline.bestx, self.rline.ploty, np.linalg.inv(transform_M), (img.shape[1], img.shape[0]))

        final_img = cv2.addWeighted(img, 1, lane, 0.3, 0)

        info_text = self.get_lane_info()

        final_img = cv2.putText(final_img, info_text, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        return final_img
    
```

Let's look at the final video output.


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

output="test.mp4"
orig_clip = VideoFileClip('project_video.mp4')

frameProcessor = FrameProcessor(mtx, dist, transform_M, (img.shape[1], img.shape[0]), 5)

lane_clip = orig_clip.fl_image(frameProcessor.process_frame)
lane_clip.write_videofile(output, audio=False)
```

    [MoviePy] >>>> Building video test.mp4
    [MoviePy] Writing video test.mp4


    
    
      0%|          | 0/1261 [00:00<?, ?it/s][A[A
    
      0%|          | 1/1261 [00:00<02:09,  9.72it/s][A[A
    
      0%|          | 2/1261 [00:00<02:11,  9.55it/s][A[A
    
      0%|          | 4/1261 [00:00<02:08,  9.78it/s][A[A
    
      0%|          | 6/1261 [00:00<02:04, 10.10it/s][A[A
    
      1%|          | 8/1261 [00:00<02:03, 10.15it/s][A[A
    
      1%|          | 9/1261 [00:00<02:04, 10.05it/s][A[A
    
      1%|          | 11/1261 [00:01<02:02, 10.18it/s][A[A
    
      1%|          | 13/1261 [00:01<02:01, 10.30it/s][A[A
    
      1%|          | 15/1261 [00:01<02:00, 10.34it/s][A[A
    
      1%|▏         | 17/1261 [00:01<02:00, 10.34it/s][A[A
    
      2%|▏         | 19/1261 [00:01<01:58, 10.47it/s][A[A
    
      2%|▏         | 21/1261 [00:02<01:58, 10.44it/s][A[A
    
      2%|▏         | 23/1261 [00:02<02:00, 10.27it/s][A[A
    
      2%|▏         | 25/1261 [00:02<02:02, 10.12it/s][A[A
    
      2%|▏         | 27/1261 [00:02<02:02, 10.05it/s][A[A
    
      2%|▏         | 29/1261 [00:02<02:02, 10.07it/s][A[A
    
      2%|▏         | 31/1261 [00:03<02:06,  9.76it/s][A[A
    
      3%|▎         | 32/1261 [00:03<02:15,  9.07it/s][A[A
    
      3%|▎         | 33/1261 [00:03<02:16,  8.97it/s][A[A
    
      3%|▎         | 34/1261 [00:03<02:18,  8.85it/s][A[A
    
      3%|▎         | 35/1261 [00:03<02:18,  8.83it/s][A[A
    
      3%|▎         | 36/1261 [00:03<02:23,  8.54it/s][A[A
    
      3%|▎         | 37/1261 [00:03<02:21,  8.66it/s][A[A
    
      3%|▎         | 38/1261 [00:03<02:18,  8.86it/s][A[A
    
      3%|▎         | 39/1261 [00:03<02:15,  9.00it/s][A[A
    
      3%|▎         | 40/1261 [00:04<02:17,  8.89it/s][A[A
    
      3%|▎         | 41/1261 [00:04<02:15,  9.03it/s][A[A
    
      3%|▎         | 42/1261 [00:04<02:14,  9.07it/s][A[A
    
      3%|▎         | 43/1261 [00:04<02:16,  8.91it/s][A[A
    
      3%|▎         | 44/1261 [00:04<02:14,  9.06it/s][A[A
    
      4%|▎         | 45/1261 [00:04<02:12,  9.17it/s][A[A
    
      4%|▎         | 46/1261 [00:04<02:11,  9.25it/s][A[A
    
      4%|▍         | 48/1261 [00:04<02:07,  9.48it/s][A[A
    
      4%|▍         | 50/1261 [00:05<02:03,  9.77it/s][A[A
    
      4%|▍         | 51/1261 [00:05<02:04,  9.72it/s][A[A
    
      4%|▍         | 52/1261 [00:05<02:04,  9.71it/s][A[A
    
      4%|▍         | 53/1261 [00:05<02:06,  9.54it/s][A[A
    
      4%|▍         | 54/1261 [00:05<02:06,  9.56it/s][A[A
    
      4%|▍         | 55/1261 [00:05<02:07,  9.45it/s][A[A
    
      4%|▍         | 56/1261 [00:05<02:09,  9.34it/s][A[A
    
      5%|▍         | 57/1261 [00:05<02:10,  9.25it/s][A[A
    
      5%|▍         | 58/1261 [00:05<02:12,  9.09it/s][A[A
    
      5%|▍         | 59/1261 [00:06<02:10,  9.18it/s][A[A
    
      5%|▍         | 60/1261 [00:06<02:11,  9.13it/s][A[A
    
      5%|▍         | 61/1261 [00:06<02:10,  9.18it/s][A[A
    
      5%|▍         | 62/1261 [00:06<02:14,  8.92it/s][A[A
    
      5%|▍         | 63/1261 [00:06<02:13,  9.00it/s][A[A
    
      5%|▌         | 64/1261 [00:06<02:13,  8.98it/s][A[A
    
      5%|▌         | 65/1261 [00:06<02:11,  9.08it/s][A[A
    
      5%|▌         | 66/1261 [00:06<02:12,  9.05it/s][A[A
    
      5%|▌         | 67/1261 [00:06<02:14,  8.88it/s][A[A
    
      5%|▌         | 68/1261 [00:07<02:15,  8.82it/s][A[A
    
      5%|▌         | 69/1261 [00:07<02:13,  8.94it/s][A[A
    
      6%|▌         | 70/1261 [00:07<02:09,  9.19it/s][A[A
    
      6%|▌         | 71/1261 [00:07<02:07,  9.33it/s][A[A
    
      6%|▌         | 72/1261 [00:07<02:05,  9.51it/s][A[A
    
      6%|▌         | 73/1261 [00:07<02:03,  9.59it/s][A[A
    
      6%|▌         | 74/1261 [00:07<02:04,  9.55it/s][A[A
    
      6%|▌         | 75/1261 [00:07<02:03,  9.59it/s][A[A
    
      6%|▌         | 76/1261 [00:07<02:02,  9.67it/s][A[A
    
      6%|▌         | 77/1261 [00:08<02:02,  9.67it/s][A[A
    
      6%|▌         | 78/1261 [00:08<02:01,  9.74it/s][A[A
    
      6%|▋         | 79/1261 [00:08<02:01,  9.75it/s][A[A
    
      6%|▋         | 80/1261 [00:08<02:00,  9.78it/s][A[A
    
      6%|▋         | 81/1261 [00:08<02:01,  9.73it/s][A[A
    
      7%|▋         | 82/1261 [00:08<02:01,  9.72it/s][A[A
    
      7%|▋         | 83/1261 [00:08<02:03,  9.58it/s][A[A
    
      7%|▋         | 85/1261 [00:08<02:00,  9.74it/s][A[A
    
      7%|▋         | 87/1261 [00:09<01:59,  9.84it/s][A[A
    
      7%|▋         | 89/1261 [00:09<01:57,  9.94it/s][A[A
    
      7%|▋         | 91/1261 [00:09<01:56, 10.01it/s][A[A
    
      7%|▋         | 93/1261 [00:09<01:57,  9.97it/s][A[A
    
      7%|▋         | 94/1261 [00:09<01:57,  9.97it/s][A[A
    
      8%|▊         | 96/1261 [00:09<01:57,  9.95it/s][A[A
    
      8%|▊         | 97/1261 [00:10<01:58,  9.83it/s][A[A
    
      8%|▊         | 98/1261 [00:10<02:00,  9.65it/s][A[A
    
      8%|▊         | 99/1261 [00:10<02:02,  9.52it/s][A[A
    
      8%|▊         | 100/1261 [00:10<02:00,  9.65it/s][A[A
    
      8%|▊         | 101/1261 [00:10<02:00,  9.63it/s][A[A
    
      8%|▊         | 102/1261 [00:10<02:04,  9.33it/s][A[A
    
      8%|▊         | 103/1261 [00:10<02:05,  9.25it/s][A[A
    
      8%|▊         | 104/1261 [00:10<02:04,  9.30it/s][A[A
    
      8%|▊         | 105/1261 [00:10<02:03,  9.33it/s][A[A
    
      8%|▊         | 106/1261 [00:11<02:04,  9.29it/s][A[A
    
      8%|▊         | 107/1261 [00:11<02:04,  9.26it/s][A[A
    
      9%|▊         | 108/1261 [00:11<02:05,  9.18it/s][A[A
    
      9%|▊         | 109/1261 [00:11<02:06,  9.09it/s][A[A
    
      9%|▊         | 110/1261 [00:11<02:08,  8.97it/s][A[A
    
      9%|▉         | 111/1261 [00:11<02:06,  9.10it/s][A[A
    
      9%|▉         | 112/1261 [00:11<02:05,  9.12it/s][A[A
    
      9%|▉         | 113/1261 [00:11<02:03,  9.32it/s][A[A
    
      9%|▉         | 115/1261 [00:11<02:00,  9.52it/s][A[A
    
      9%|▉         | 116/1261 [00:12<01:58,  9.64it/s][A[A
    
      9%|▉         | 117/1261 [00:12<01:57,  9.73it/s][A[A
    
      9%|▉         | 118/1261 [00:12<02:00,  9.51it/s][A[A
    
      9%|▉         | 119/1261 [00:12<02:00,  9.46it/s][A[A
    
     10%|▉         | 120/1261 [00:12<02:02,  9.29it/s][A[A
    
     10%|▉         | 121/1261 [00:12<02:02,  9.29it/s][A[A
    
     10%|▉         | 122/1261 [00:12<02:05,  9.11it/s][A[A
    
     10%|▉         | 123/1261 [00:12<02:03,  9.21it/s][A[A
    
     10%|▉         | 124/1261 [00:12<02:03,  9.18it/s][A[A
    
     10%|▉         | 125/1261 [00:13<02:04,  9.16it/s][A[A
    
     10%|▉         | 126/1261 [00:13<02:03,  9.17it/s][A[A
    
     10%|█         | 127/1261 [00:13<02:01,  9.34it/s][A[A
    
     10%|█         | 128/1261 [00:13<02:01,  9.31it/s][A[A
    
     10%|█         | 129/1261 [00:13<02:00,  9.42it/s][A[A
    
     10%|█         | 130/1261 [00:13<01:58,  9.53it/s][A[A
    
     10%|█         | 131/1261 [00:13<01:59,  9.49it/s][A[A
    
     10%|█         | 132/1261 [00:13<01:58,  9.51it/s][A[A
    
     11%|█         | 134/1261 [00:14<01:57,  9.61it/s][A[A
    
     11%|█         | 135/1261 [00:14<01:59,  9.46it/s][A[A
    
     11%|█         | 136/1261 [00:14<01:57,  9.61it/s][A[A
    
     11%|█         | 137/1261 [00:14<01:56,  9.61it/s][A[A
    
     11%|█         | 138/1261 [00:14<01:57,  9.54it/s][A[A
    
     11%|█         | 139/1261 [00:14<01:58,  9.47it/s][A[A
    
     11%|█         | 140/1261 [00:14<01:57,  9.57it/s][A[A
    
     11%|█         | 141/1261 [00:14<01:57,  9.55it/s][A[A
    
     11%|█▏        | 142/1261 [00:14<01:58,  9.46it/s][A[A
    
     11%|█▏        | 143/1261 [00:14<01:57,  9.51it/s][A[A
    
     11%|█▏        | 144/1261 [00:15<01:56,  9.57it/s][A[A
    
     11%|█▏        | 145/1261 [00:15<01:57,  9.53it/s][A[A
    
     12%|█▏        | 147/1261 [00:15<01:55,  9.65it/s][A[A
    
     12%|█▏        | 148/1261 [00:15<01:54,  9.68it/s][A[A
    
     12%|█▏        | 149/1261 [00:15<01:55,  9.60it/s][A[A
    
     12%|█▏        | 150/1261 [00:15<01:56,  9.50it/s][A[A
    
     12%|█▏        | 151/1261 [00:15<01:58,  9.41it/s][A[A
    
     12%|█▏        | 152/1261 [00:15<01:57,  9.44it/s][A[A
    
     12%|█▏        | 153/1261 [00:16<01:58,  9.37it/s][A[A
    
     12%|█▏        | 154/1261 [00:16<01:57,  9.44it/s][A[A
    
     12%|█▏        | 155/1261 [00:16<01:58,  9.35it/s][A[A
    
     12%|█▏        | 156/1261 [00:16<01:57,  9.44it/s][A[A
    
     12%|█▏        | 157/1261 [00:16<01:58,  9.32it/s][A[A
    
     13%|█▎        | 158/1261 [00:16<01:58,  9.34it/s][A[A
    
     13%|█▎        | 159/1261 [00:16<01:57,  9.41it/s][A[A
    
     13%|█▎        | 160/1261 [00:16<01:57,  9.35it/s][A[A
    
     13%|█▎        | 161/1261 [00:16<01:58,  9.31it/s][A[A
    
     13%|█▎        | 162/1261 [00:16<01:59,  9.20it/s][A[A
    
     13%|█▎        | 163/1261 [00:17<01:58,  9.27it/s][A[A
    
     13%|█▎        | 164/1261 [00:17<01:59,  9.17it/s][A[A
    
     13%|█▎        | 165/1261 [00:17<02:00,  9.13it/s][A[A
    
     13%|█▎        | 166/1261 [00:17<02:01,  9.01it/s][A[A
    
     13%|█▎        | 167/1261 [00:17<02:03,  8.87it/s][A[A
    
     13%|█▎        | 168/1261 [00:17<02:01,  8.99it/s][A[A
    
     13%|█▎        | 169/1261 [00:17<02:02,  8.94it/s][A[A
    
     13%|█▎        | 170/1261 [00:17<02:01,  8.97it/s][A[A
    
     14%|█▎        | 171/1261 [00:17<02:01,  8.94it/s][A[A
    
     14%|█▎        | 172/1261 [00:18<02:01,  8.96it/s][A[A
    
     14%|█▎        | 173/1261 [00:18<02:00,  9.02it/s][A[A
    
     14%|█▍        | 174/1261 [00:18<01:59,  9.13it/s][A[A
    
     14%|█▍        | 175/1261 [00:18<01:59,  9.09it/s][A[A
    
     14%|█▍        | 176/1261 [00:18<01:58,  9.17it/s][A[A
    
     14%|█▍        | 177/1261 [00:18<01:56,  9.30it/s][A[A
    
     14%|█▍        | 178/1261 [00:18<02:00,  8.96it/s][A[A
    
     14%|█▍        | 179/1261 [00:18<02:00,  8.98it/s][A[A
    
     14%|█▍        | 180/1261 [00:18<02:01,  8.92it/s][A[A
    
     14%|█▍        | 181/1261 [00:19<02:02,  8.80it/s][A[A
    
     14%|█▍        | 182/1261 [00:19<02:06,  8.51it/s][A[A
    
     15%|█▍        | 183/1261 [00:19<02:08,  8.42it/s][A[A
    
     15%|█▍        | 184/1261 [00:19<02:07,  8.46it/s][A[A
    
     15%|█▍        | 185/1261 [00:19<02:03,  8.73it/s][A[A
    
     15%|█▍        | 186/1261 [00:19<02:01,  8.85it/s][A[A
    
     15%|█▍        | 187/1261 [00:19<02:01,  8.84it/s][A[A
    
     15%|█▍        | 188/1261 [00:19<02:02,  8.73it/s][A[A
    
     15%|█▍        | 189/1261 [00:20<01:59,  8.94it/s][A[A
    
     15%|█▌        | 190/1261 [00:20<01:58,  9.04it/s][A[A
    
     15%|█▌        | 191/1261 [00:20<01:57,  9.10it/s][A[A
    
     15%|█▌        | 192/1261 [00:20<01:59,  8.97it/s][A[A
    
     15%|█▌        | 193/1261 [00:20<02:00,  8.83it/s][A[A
    
     15%|█▌        | 194/1261 [00:20<02:00,  8.85it/s][A[A
    
     15%|█▌        | 195/1261 [00:20<02:00,  8.86it/s][A[A
    
     16%|█▌        | 196/1261 [00:20<02:00,  8.81it/s][A[A
    
     16%|█▌        | 197/1261 [00:20<02:01,  8.74it/s][A[A
    
     16%|█▌        | 198/1261 [00:21<02:00,  8.79it/s][A[A
    
     16%|█▌        | 199/1261 [00:21<02:03,  8.58it/s][A[A
    
     16%|█▌        | 200/1261 [00:21<02:03,  8.58it/s][A[A
    
     16%|█▌        | 201/1261 [00:21<02:01,  8.75it/s][A[A
    
     16%|█▌        | 202/1261 [00:21<01:57,  9.01it/s][A[A
    
     16%|█▌        | 203/1261 [00:21<01:54,  9.22it/s][A[A
    
     16%|█▌        | 204/1261 [00:21<01:53,  9.30it/s][A[A
    
     16%|█▋        | 205/1261 [00:21<01:57,  8.97it/s][A[A
    
     16%|█▋        | 206/1261 [00:21<01:57,  9.00it/s][A[A
    
     16%|█▋        | 207/1261 [00:22<01:57,  8.97it/s][A[A
    
     16%|█▋        | 208/1261 [00:22<01:55,  9.10it/s][A[A
    
     17%|█▋        | 209/1261 [00:22<01:56,  8.99it/s][A[A
    
     17%|█▋        | 210/1261 [00:22<01:55,  9.07it/s][A[A
    
     17%|█▋        | 211/1261 [00:22<01:53,  9.24it/s][A[A
    
     17%|█▋        | 212/1261 [00:22<01:53,  9.24it/s][A[A
    
     17%|█▋        | 213/1261 [00:22<01:54,  9.16it/s][A[A
    
     17%|█▋        | 214/1261 [00:22<01:54,  9.14it/s][A[A
    
     17%|█▋        | 215/1261 [00:22<01:52,  9.31it/s][A[A
    
     17%|█▋        | 216/1261 [00:22<01:51,  9.39it/s][A[A
    
     17%|█▋        | 217/1261 [00:23<01:51,  9.38it/s][A[A
    
     17%|█▋        | 218/1261 [00:23<01:53,  9.17it/s][A[A
    
     17%|█▋        | 219/1261 [00:23<01:52,  9.25it/s][A[A
    
     17%|█▋        | 220/1261 [00:23<01:53,  9.15it/s][A[A
    
     18%|█▊        | 221/1261 [00:23<02:02,  8.49it/s][A[A
    
     18%|█▊        | 222/1261 [00:23<02:01,  8.57it/s][A[A
    
     18%|█▊        | 223/1261 [00:23<01:59,  8.68it/s][A[A
    
     18%|█▊        | 224/1261 [00:23<01:58,  8.76it/s][A[A
    
     18%|█▊        | 225/1261 [00:24<01:58,  8.72it/s][A[A
    
     18%|█▊        | 226/1261 [00:24<01:56,  8.87it/s][A[A
    
     18%|█▊        | 227/1261 [00:24<01:57,  8.81it/s][A[A
    
     18%|█▊        | 228/1261 [00:24<01:55,  8.95it/s][A[A
    
     18%|█▊        | 229/1261 [00:24<01:55,  8.96it/s][A[A
    
     18%|█▊        | 230/1261 [00:24<01:54,  9.03it/s][A[A
    
     18%|█▊        | 231/1261 [00:24<01:54,  9.00it/s][A[A
    
     18%|█▊        | 232/1261 [00:24<01:58,  8.71it/s][A[A
    
     18%|█▊        | 233/1261 [00:24<01:56,  8.83it/s][A[A
    
     19%|█▊        | 234/1261 [00:25<01:57,  8.76it/s][A[A
    
     19%|█▊        | 235/1261 [00:25<01:53,  9.00it/s][A[A
    
     19%|█▊        | 236/1261 [00:25<01:54,  8.97it/s][A[A
    
     19%|█▉        | 237/1261 [00:25<01:51,  9.15it/s][A[A
    
     19%|█▉        | 238/1261 [00:25<01:53,  9.00it/s][A[A
    
     19%|█▉        | 239/1261 [00:25<01:54,  8.93it/s][A[A
    
     19%|█▉        | 240/1261 [00:25<01:53,  9.02it/s][A[A
    
     19%|█▉        | 241/1261 [00:25<01:52,  9.07it/s][A[A
    
     19%|█▉        | 242/1261 [00:25<01:53,  9.00it/s][A[A
    
     19%|█▉        | 243/1261 [00:26<01:56,  8.75it/s][A[A
    
     19%|█▉        | 244/1261 [00:26<01:58,  8.62it/s][A[A
    
     19%|█▉        | 245/1261 [00:26<01:59,  8.52it/s][A[A
    
     20%|█▉        | 246/1261 [00:26<01:54,  8.84it/s][A[A
    
     20%|█▉        | 247/1261 [00:26<01:50,  9.15it/s][A[A
    
     20%|█▉        | 249/1261 [00:26<01:46,  9.51it/s][A[A
    
     20%|█▉        | 250/1261 [00:26<01:44,  9.64it/s][A[A
    
     20%|█▉        | 252/1261 [00:26<01:42,  9.84it/s][A[A
    
     20%|██        | 253/1261 [00:27<01:42,  9.84it/s][A[A
    
     20%|██        | 255/1261 [00:27<01:40,  9.96it/s][A[A
    
     20%|██        | 256/1261 [00:27<01:41,  9.89it/s][A[A
    
     20%|██        | 258/1261 [00:27<01:40,  9.99it/s][A[A
    
     21%|██        | 260/1261 [00:27<01:40,  9.96it/s][A[A
    
     21%|██        | 262/1261 [00:27<01:38, 10.14it/s][A[A
    
     21%|██        | 264/1261 [00:28<01:37, 10.24it/s][A[A
    
     21%|██        | 266/1261 [00:28<01:37, 10.24it/s][A[A
    
     21%|██▏       | 268/1261 [00:28<01:39,  9.99it/s][A[A
    
     21%|██▏       | 270/1261 [00:28<01:37, 10.12it/s][A[A
    
     22%|██▏       | 272/1261 [00:28<01:37, 10.15it/s][A[A
    
     22%|██▏       | 274/1261 [00:29<01:36, 10.22it/s][A[A
    
     22%|██▏       | 276/1261 [00:29<01:36, 10.21it/s][A[A
    
     22%|██▏       | 278/1261 [00:29<01:36, 10.23it/s][A[A
    
     22%|██▏       | 280/1261 [00:29<01:35, 10.32it/s][A[A
    
     22%|██▏       | 282/1261 [00:29<01:35, 10.26it/s][A[A
    
     23%|██▎       | 284/1261 [00:30<01:35, 10.25it/s][A[A
    
     23%|██▎       | 286/1261 [00:30<01:36, 10.15it/s][A[A
    
     23%|██▎       | 288/1261 [00:30<01:38,  9.91it/s][A[A
    
     23%|██▎       | 289/1261 [00:30<01:43,  9.37it/s][A[A
    
     23%|██▎       | 291/1261 [00:30<01:40,  9.63it/s][A[A
    
     23%|██▎       | 292/1261 [00:30<01:45,  9.18it/s][A[A
    
     23%|██▎       | 293/1261 [00:31<01:43,  9.34it/s][A[A
    
     23%|██▎       | 295/1261 [00:31<01:40,  9.64it/s][A[A
    
     24%|██▎       | 297/1261 [00:31<01:37,  9.88it/s][A[A
    
     24%|██▎       | 299/1261 [00:31<01:35, 10.04it/s][A[A
    
     24%|██▍       | 301/1261 [00:31<01:37,  9.89it/s][A[A
    
     24%|██▍       | 303/1261 [00:32<01:35, 10.00it/s][A[A
    
     24%|██▍       | 305/1261 [00:32<01:35, 10.04it/s][A[A
    
     24%|██▍       | 307/1261 [00:32<01:35, 10.01it/s][A[A
    
     25%|██▍       | 309/1261 [00:32<01:34, 10.10it/s][A[A
    
     25%|██▍       | 311/1261 [00:32<01:33, 10.11it/s][A[A
    
     25%|██▍       | 313/1261 [00:33<01:33, 10.17it/s][A[A
    
     25%|██▍       | 315/1261 [00:33<01:33, 10.14it/s][A[A
    
     25%|██▌       | 317/1261 [00:33<01:32, 10.20it/s][A[A
    
     25%|██▌       | 319/1261 [00:33<01:31, 10.27it/s][A[A
    
     25%|██▌       | 321/1261 [00:33<01:31, 10.27it/s][A[A
    
     26%|██▌       | 323/1261 [00:33<01:31, 10.24it/s][A[A
    
     26%|██▌       | 325/1261 [00:34<01:31, 10.25it/s][A[A
    
     26%|██▌       | 327/1261 [00:34<01:30, 10.27it/s][A[A
    
     26%|██▌       | 329/1261 [00:34<01:30, 10.32it/s][A[A
    
     26%|██▌       | 331/1261 [00:34<01:29, 10.34it/s][A[A
    
     26%|██▋       | 333/1261 [00:34<01:30, 10.30it/s][A[A
    
     27%|██▋       | 335/1261 [00:35<01:29, 10.33it/s][A[A
    
     27%|██▋       | 337/1261 [00:35<01:29, 10.31it/s][A[A
    
     27%|██▋       | 339/1261 [00:35<01:29, 10.30it/s][A[A
    
     27%|██▋       | 341/1261 [00:35<01:29, 10.33it/s][A[A
    
     27%|██▋       | 343/1261 [00:35<01:29, 10.29it/s][A[A
    
     27%|██▋       | 345/1261 [00:36<01:29, 10.24it/s][A[A
    
     28%|██▊       | 347/1261 [00:36<01:29, 10.18it/s][A[A
    
     28%|██▊       | 349/1261 [00:36<01:30, 10.12it/s][A[A
    
     28%|██▊       | 351/1261 [00:36<01:30, 10.03it/s][A[A
    
     28%|██▊       | 353/1261 [00:36<01:30,  9.99it/s][A[A
    
     28%|██▊       | 355/1261 [00:37<01:30, 10.06it/s][A[A
    
     28%|██▊       | 357/1261 [00:37<01:30, 10.02it/s][A[A
    
     28%|██▊       | 359/1261 [00:37<01:28, 10.14it/s][A[A
    
     29%|██▊       | 361/1261 [00:37<01:28, 10.20it/s][A[A
    
     29%|██▉       | 363/1261 [00:37<01:27, 10.28it/s][A[A
    
     29%|██▉       | 365/1261 [00:38<01:26, 10.32it/s][A[A
    
     29%|██▉       | 367/1261 [00:38<01:27, 10.27it/s][A[A
    
     29%|██▉       | 369/1261 [00:38<01:27, 10.25it/s][A[A
    
     29%|██▉       | 371/1261 [00:38<01:26, 10.27it/s][A[A
    
     30%|██▉       | 373/1261 [00:38<01:26, 10.32it/s][A[A
    
     30%|██▉       | 375/1261 [00:39<01:25, 10.39it/s][A[A
    
     30%|██▉       | 377/1261 [00:39<01:26, 10.18it/s][A[A
    
     30%|███       | 379/1261 [00:39<01:27, 10.05it/s][A[A
    
     30%|███       | 381/1261 [00:39<01:27, 10.04it/s][A[A
    
     30%|███       | 383/1261 [00:39<01:27, 10.07it/s][A[A
    
     31%|███       | 385/1261 [00:40<01:26, 10.13it/s][A[A
    
     31%|███       | 387/1261 [00:40<01:26, 10.14it/s][A[A
    
     31%|███       | 389/1261 [00:40<01:26, 10.11it/s][A[A
    
     31%|███       | 391/1261 [00:40<01:25, 10.21it/s][A[A
    
     31%|███       | 393/1261 [00:40<01:24, 10.26it/s][A[A
    
     31%|███▏      | 395/1261 [00:41<01:24, 10.28it/s][A[A
    
     31%|███▏      | 397/1261 [00:41<01:23, 10.33it/s][A[A
    
     32%|███▏      | 399/1261 [00:41<01:24, 10.16it/s][A[A
    
     32%|███▏      | 401/1261 [00:41<01:24, 10.16it/s][A[A
    
     32%|███▏      | 403/1261 [00:41<01:24, 10.15it/s][A[A
    
     32%|███▏      | 405/1261 [00:42<01:24, 10.19it/s][A[A
    
     32%|███▏      | 407/1261 [00:42<01:25, 10.01it/s][A[A
    
     32%|███▏      | 409/1261 [00:42<01:25,  9.97it/s][A[A
    
     33%|███▎      | 410/1261 [00:42<01:25,  9.90it/s][A[A
    
     33%|███▎      | 412/1261 [00:42<01:25,  9.99it/s][A[A
    
     33%|███▎      | 413/1261 [00:42<01:25,  9.94it/s][A[A
    
     33%|███▎      | 414/1261 [00:42<01:25,  9.92it/s][A[A
    
     33%|███▎      | 416/1261 [00:43<01:24, 10.00it/s][A[A
    
     33%|███▎      | 418/1261 [00:43<01:23, 10.05it/s][A[A
    
     33%|███▎      | 420/1261 [00:43<01:24,  9.97it/s][A[A
    
     33%|███▎      | 422/1261 [00:43<01:23, 10.05it/s][A[A
    
     34%|███▎      | 424/1261 [00:43<01:24,  9.93it/s][A[A
    
     34%|███▎      | 425/1261 [00:44<01:25,  9.82it/s][A[A
    
     34%|███▍      | 426/1261 [00:44<01:25,  9.79it/s][A[A
    
     34%|███▍      | 428/1261 [00:44<01:24,  9.91it/s][A[A
    
     34%|███▍      | 430/1261 [00:44<01:23, 10.01it/s][A[A
    
     34%|███▍      | 432/1261 [00:44<01:22, 10.02it/s][A[A
    
     34%|███▍      | 434/1261 [00:44<01:21, 10.10it/s][A[A
    
     35%|███▍      | 436/1261 [00:45<01:22, 10.04it/s][A[A
    
     35%|███▍      | 438/1261 [00:45<01:22,  9.96it/s][A[A
    
     35%|███▍      | 439/1261 [00:45<01:22,  9.93it/s][A[A
    
     35%|███▍      | 441/1261 [00:45<01:21, 10.06it/s][A[A
    
     35%|███▌      | 443/1261 [00:45<01:21, 10.06it/s][A[A
    
     35%|███▌      | 445/1261 [00:46<01:21, 10.06it/s][A[A
    
     35%|███▌      | 447/1261 [00:46<01:20, 10.08it/s][A[A
    
     36%|███▌      | 449/1261 [00:46<01:21,  9.96it/s][A[A
    
     36%|███▌      | 451/1261 [00:46<01:20, 10.04it/s][A[A
    
     36%|███▌      | 453/1261 [00:46<01:19, 10.11it/s][A[A
    
     36%|███▌      | 455/1261 [00:47<01:20,  9.99it/s][A[A
    
     36%|███▌      | 457/1261 [00:47<01:20, 10.00it/s][A[A
    
     36%|███▋      | 459/1261 [00:47<01:20,  9.92it/s][A[A
    
     36%|███▋      | 460/1261 [00:47<01:21,  9.83it/s][A[A
    
     37%|███▋      | 461/1261 [00:47<01:21,  9.84it/s][A[A
    
     37%|███▋      | 463/1261 [00:47<01:20,  9.97it/s][A[A
    
     37%|███▋      | 465/1261 [00:48<01:18, 10.14it/s][A[A
    
     37%|███▋      | 467/1261 [00:48<01:18, 10.07it/s][A[A
    
     37%|███▋      | 469/1261 [00:48<01:19,  9.99it/s][A[A
    
     37%|███▋      | 471/1261 [00:48<01:19,  9.89it/s][A[A
    
     37%|███▋      | 472/1261 [00:48<01:21,  9.74it/s][A[A
    
     38%|███▊      | 473/1261 [00:48<01:21,  9.73it/s][A[A
    
     38%|███▊      | 475/1261 [00:49<01:19,  9.84it/s][A[A
    
     38%|███▊      | 476/1261 [00:49<01:19,  9.83it/s][A[A
    
     38%|███▊      | 477/1261 [00:49<01:19,  9.81it/s][A[A
    
     38%|███▊      | 478/1261 [00:49<01:20,  9.75it/s][A[A
    
     38%|███▊      | 479/1261 [00:49<01:20,  9.76it/s][A[A
    
     38%|███▊      | 480/1261 [00:49<01:19,  9.79it/s][A[A
    
     38%|███▊      | 481/1261 [00:49<01:19,  9.76it/s][A[A
    
     38%|███▊      | 482/1261 [00:49<01:19,  9.74it/s][A[A
    
     38%|███▊      | 483/1261 [00:49<01:19,  9.79it/s][A[A
    
     38%|███▊      | 485/1261 [00:50<01:18,  9.83it/s][A[A
    
     39%|███▊      | 487/1261 [00:50<01:17,  9.97it/s][A[A
    
     39%|███▉      | 489/1261 [00:50<01:16, 10.11it/s][A[A
    
     39%|███▉      | 491/1261 [00:50<01:15, 10.15it/s][A[A
    
     39%|███▉      | 493/1261 [00:50<01:15, 10.17it/s][A[A
    
     39%|███▉      | 495/1261 [00:51<01:16,  9.98it/s][A[A
    
     39%|███▉      | 496/1261 [00:51<01:18,  9.78it/s][A[A
    
     39%|███▉      | 498/1261 [00:51<01:17,  9.88it/s][A[A
    
     40%|███▉      | 500/1261 [00:51<01:15, 10.04it/s][A[A
    
     40%|███▉      | 502/1261 [00:51<01:15, 10.12it/s][A[A
    
     40%|███▉      | 504/1261 [00:51<01:15, 10.07it/s][A[A
    
     40%|████      | 506/1261 [00:52<01:14, 10.12it/s][A[A
    
     40%|████      | 508/1261 [00:52<01:14, 10.11it/s][A[A
    
     40%|████      | 510/1261 [00:52<01:13, 10.16it/s][A[A
    
     41%|████      | 512/1261 [00:52<01:14, 10.06it/s][A[A
    
     41%|████      | 514/1261 [00:52<01:14,  9.98it/s][A[A
    
     41%|████      | 516/1261 [00:53<01:14, 10.05it/s][A[A
    
     41%|████      | 518/1261 [00:53<01:13, 10.08it/s][A[A
    
     41%|████      | 520/1261 [00:53<01:13, 10.08it/s][A[A
    
     41%|████▏     | 522/1261 [00:53<01:14,  9.97it/s][A[A
    
     41%|████▏     | 523/1261 [00:53<01:14,  9.91it/s][A[A
    
     42%|████▏     | 524/1261 [00:53<01:14,  9.88it/s][A[A
    
     42%|████▏     | 526/1261 [00:54<01:13, 10.03it/s][A[A
    
     42%|████▏     | 528/1261 [00:54<01:12, 10.11it/s][A[A
    
     42%|████▏     | 530/1261 [00:54<01:13,  9.94it/s][A[A
    
     42%|████▏     | 531/1261 [00:54<01:13,  9.88it/s][A[A
    
     42%|████▏     | 532/1261 [00:54<01:15,  9.70it/s][A[A
    
     42%|████▏     | 533/1261 [00:54<01:15,  9.68it/s][A[A
    
     42%|████▏     | 534/1261 [00:54<01:15,  9.69it/s][A[A
    
     42%|████▏     | 535/1261 [00:55<01:15,  9.64it/s][A[A
    
     43%|████▎     | 536/1261 [00:55<01:16,  9.51it/s][A[A
    
     43%|████▎     | 538/1261 [00:55<01:14,  9.67it/s][A[A
    
     43%|████▎     | 540/1261 [00:55<01:11, 10.02it/s][A[A
    
     43%|████▎     | 542/1261 [00:55<01:09, 10.31it/s][A[A
    
     43%|████▎     | 544/1261 [00:55<01:08, 10.47it/s][A[A
    
     43%|████▎     | 546/1261 [00:56<01:07, 10.62it/s][A[A
    
     43%|████▎     | 548/1261 [00:56<01:06, 10.68it/s][A[A
    
     44%|████▎     | 550/1261 [00:56<01:06, 10.72it/s][A[A
    
     44%|████▍     | 552/1261 [00:56<01:04, 10.94it/s][A[A
    
     44%|████▍     | 554/1261 [00:56<01:03, 11.06it/s][A[A
    
     44%|████▍     | 556/1261 [00:57<01:04, 10.95it/s][A[A
    
     44%|████▍     | 558/1261 [00:57<01:04, 10.95it/s][A[A
    
     44%|████▍     | 560/1261 [00:57<01:04, 10.92it/s][A[A
    
     45%|████▍     | 562/1261 [00:57<01:03, 11.08it/s][A[A
    
     45%|████▍     | 564/1261 [00:57<01:01, 11.26it/s][A[A
    
     45%|████▍     | 566/1261 [00:57<01:01, 11.35it/s][A[A
    
     45%|████▌     | 568/1261 [00:58<01:00, 11.38it/s][A[A
    
     45%|████▌     | 570/1261 [00:58<01:01, 11.17it/s][A[A
    
     45%|████▌     | 572/1261 [00:58<01:01, 11.12it/s][A[A
    
     46%|████▌     | 574/1261 [00:58<01:01, 11.22it/s][A[A
    
     46%|████▌     | 576/1261 [00:58<01:02, 11.00it/s][A[A
    
     46%|████▌     | 578/1261 [00:58<01:04, 10.64it/s][A[A
    
     46%|████▌     | 580/1261 [00:59<01:05, 10.43it/s][A[A
    
     46%|████▌     | 582/1261 [00:59<01:06, 10.24it/s][A[A
    
     46%|████▋     | 584/1261 [00:59<01:05, 10.35it/s][A[A
    
     46%|████▋     | 586/1261 [00:59<01:04, 10.51it/s][A[A
    
     47%|████▋     | 588/1261 [00:59<01:04, 10.38it/s][A[A
    
     47%|████▋     | 590/1261 [01:00<01:04, 10.34it/s][A[A
    
     47%|████▋     | 592/1261 [01:00<01:03, 10.56it/s][A[A
    
     47%|████▋     | 594/1261 [01:00<01:01, 10.89it/s][A[A
    
     47%|████▋     | 596/1261 [01:00<00:59, 11.21it/s][A[A
    
     47%|████▋     | 598/1261 [01:00<00:59, 11.15it/s][A[A
    
     48%|████▊     | 600/1261 [01:01<00:58, 11.31it/s][A[A
    
     48%|████▊     | 602/1261 [01:01<00:57, 11.46it/s][A[A
    
     48%|████▊     | 604/1261 [01:01<00:57, 11.47it/s][A[A
    
     48%|████▊     | 606/1261 [01:01<00:56, 11.59it/s][A[A
    
     48%|████▊     | 608/1261 [01:01<00:56, 11.53it/s][A[A
    
     48%|████▊     | 610/1261 [01:01<00:56, 11.47it/s][A[A
    
     49%|████▊     | 612/1261 [01:02<00:57, 11.29it/s][A[A
    
     49%|████▊     | 614/1261 [01:02<00:59, 10.92it/s][A[A
    
     49%|████▉     | 616/1261 [01:02<01:00, 10.69it/s][A[A
    
     49%|████▉     | 618/1261 [01:02<01:02, 10.29it/s][A[A
    
     49%|████▉     | 620/1261 [01:02<01:02, 10.32it/s][A[A
    
     49%|████▉     | 622/1261 [01:03<01:01, 10.41it/s][A[A
    
     49%|████▉     | 624/1261 [01:03<01:03,  9.98it/s][A[A
    
     50%|████▉     | 626/1261 [01:03<01:04,  9.82it/s][A[A
    
     50%|████▉     | 627/1261 [01:03<01:05,  9.74it/s][A[A
    
     50%|████▉     | 628/1261 [01:03<01:07,  9.32it/s][A[A
    
     50%|████▉     | 630/1261 [01:03<01:05,  9.69it/s][A[A
    
     50%|█████     | 632/1261 [01:04<01:03,  9.87it/s][A[A
    
     50%|█████     | 633/1261 [01:04<01:04,  9.72it/s][A[A
    
     50%|█████     | 634/1261 [01:04<01:04,  9.78it/s][A[A
    
     50%|█████     | 635/1261 [01:04<01:04,  9.74it/s][A[A
    
     50%|█████     | 636/1261 [01:04<01:04,  9.66it/s][A[A
    
     51%|█████     | 638/1261 [01:04<01:04,  9.69it/s][A[A
    
     51%|█████     | 639/1261 [01:04<01:05,  9.51it/s][A[A
    
     51%|█████     | 640/1261 [01:04<01:06,  9.29it/s][A[A
    
     51%|█████     | 641/1261 [01:05<01:15,  8.20it/s][A[A
    
     51%|█████     | 642/1261 [01:05<01:15,  8.17it/s][A[A
    
     51%|█████     | 643/1261 [01:05<01:16,  8.07it/s][A[A
    
     51%|█████     | 644/1261 [01:05<01:17,  7.97it/s][A[A
    
     51%|█████     | 645/1261 [01:05<01:13,  8.38it/s][A[A
    
     51%|█████     | 646/1261 [01:05<01:10,  8.67it/s][A[A
    
     51%|█████▏    | 647/1261 [01:05<01:09,  8.85it/s][A[A
    
     51%|█████▏    | 648/1261 [01:05<01:09,  8.87it/s][A[A
    
     51%|█████▏    | 649/1261 [01:06<01:10,  8.73it/s][A[A
    
     52%|█████▏    | 650/1261 [01:06<01:11,  8.52it/s][A[A
    
     52%|█████▏    | 651/1261 [01:06<01:11,  8.56it/s][A[A
    
     52%|█████▏    | 652/1261 [01:06<01:10,  8.58it/s][A[A
    
     52%|█████▏    | 653/1261 [01:06<01:10,  8.58it/s][A[A
    
     52%|█████▏    | 654/1261 [01:06<01:10,  8.61it/s][A[A
    
     52%|█████▏    | 655/1261 [01:06<01:10,  8.62it/s][A[A
    
     52%|█████▏    | 656/1261 [01:06<01:10,  8.57it/s][A[A
    
     52%|█████▏    | 657/1261 [01:06<01:11,  8.49it/s][A[A
    
     52%|█████▏    | 658/1261 [01:07<01:10,  8.60it/s][A[A
    
     52%|█████▏    | 659/1261 [01:07<01:09,  8.64it/s][A[A
    
     52%|█████▏    | 660/1261 [01:07<01:11,  8.38it/s][A[A
    
     52%|█████▏    | 661/1261 [01:07<01:10,  8.50it/s][A[A
    
     52%|█████▏    | 662/1261 [01:07<01:07,  8.82it/s][A[A
    
     53%|█████▎    | 663/1261 [01:07<01:06,  9.04it/s][A[A
    
     53%|█████▎    | 664/1261 [01:07<01:05,  9.13it/s][A[A
    
     53%|█████▎    | 665/1261 [01:07<01:04,  9.29it/s][A[A
    
     53%|█████▎    | 666/1261 [01:07<01:05,  9.02it/s][A[A
    
     53%|█████▎    | 667/1261 [01:08<01:05,  9.02it/s][A[A
    
     53%|█████▎    | 668/1261 [01:08<01:04,  9.20it/s][A[A
    
     53%|█████▎    | 669/1261 [01:08<01:03,  9.35it/s][A[A
    
     53%|█████▎    | 670/1261 [01:08<01:04,  9.22it/s][A[A
    
     53%|█████▎    | 671/1261 [01:08<01:04,  9.22it/s][A[A
    
     53%|█████▎    | 672/1261 [01:08<01:02,  9.39it/s][A[A
    
     53%|█████▎    | 673/1261 [01:08<01:03,  9.24it/s][A[A
    
     54%|█████▎    | 675/1261 [01:08<01:02,  9.31it/s][A[A
    
     54%|█████▎    | 676/1261 [01:09<01:03,  9.28it/s][A[A
    
     54%|█████▎    | 677/1261 [01:09<01:04,  9.07it/s][A[A
    
     54%|█████▍    | 678/1261 [01:09<01:04,  9.09it/s][A[A
    
     54%|█████▍    | 679/1261 [01:09<01:10,  8.26it/s][A[A
    
     54%|█████▍    | 680/1261 [01:09<01:09,  8.33it/s][A[A
    
     54%|█████▍    | 681/1261 [01:09<01:09,  8.36it/s][A[A
    
     54%|█████▍    | 682/1261 [01:09<01:10,  8.21it/s][A[A
    
     54%|█████▍    | 683/1261 [01:09<01:13,  7.89it/s][A[A
    
     54%|█████▍    | 684/1261 [01:10<01:11,  8.05it/s][A[A
    
     54%|█████▍    | 685/1261 [01:10<01:10,  8.23it/s][A[A
    
     54%|█████▍    | 686/1261 [01:10<01:07,  8.55it/s][A[A
    
     54%|█████▍    | 687/1261 [01:10<01:07,  8.55it/s][A[A
    
     55%|█████▍    | 688/1261 [01:10<01:06,  8.65it/s][A[A
    
     55%|█████▍    | 689/1261 [01:10<01:04,  8.81it/s][A[A
    
     55%|█████▍    | 690/1261 [01:10<01:04,  8.79it/s][A[A
    
     55%|█████▍    | 691/1261 [01:10<01:03,  8.99it/s][A[A
    
     55%|█████▍    | 692/1261 [01:10<01:01,  9.25it/s][A[A
    
     55%|█████▍    | 693/1261 [01:11<01:03,  8.97it/s][A[A
    
     55%|█████▌    | 694/1261 [01:11<01:04,  8.79it/s][A[A
    
     55%|█████▌    | 695/1261 [01:11<01:02,  8.99it/s][A[A
    
     55%|█████▌    | 696/1261 [01:11<01:02,  9.09it/s][A[A
    
     55%|█████▌    | 697/1261 [01:11<01:02,  9.04it/s][A[A
    
     55%|█████▌    | 698/1261 [01:11<01:03,  8.88it/s][A[A
    
     55%|█████▌    | 699/1261 [01:11<01:02,  8.98it/s][A[A
    
     56%|█████▌    | 700/1261 [01:11<01:01,  9.15it/s][A[A
    
     56%|█████▌    | 701/1261 [01:11<00:59,  9.37it/s][A[A
    
     56%|█████▌    | 702/1261 [01:12<01:00,  9.24it/s][A[A
    
     56%|█████▌    | 703/1261 [01:12<00:59,  9.34it/s][A[A
    
     56%|█████▌    | 704/1261 [01:12<00:59,  9.32it/s][A[A
    
     56%|█████▌    | 705/1261 [01:12<01:01,  9.09it/s][A[A
    
     56%|█████▌    | 706/1261 [01:12<01:02,  8.84it/s][A[A
    
     56%|█████▌    | 707/1261 [01:12<01:02,  8.83it/s][A[A
    
     56%|█████▌    | 708/1261 [01:12<01:01,  8.92it/s][A[A
    
     56%|█████▌    | 709/1261 [01:12<01:01,  8.93it/s][A[A
    
     56%|█████▋    | 710/1261 [01:12<01:01,  8.96it/s][A[A
    
     56%|█████▋    | 711/1261 [01:13<01:00,  9.07it/s][A[A
    
     56%|█████▋    | 712/1261 [01:13<01:02,  8.81it/s][A[A
    
     57%|█████▋    | 713/1261 [01:13<01:06,  8.23it/s][A[A
    
     57%|█████▋    | 714/1261 [01:13<01:07,  8.08it/s][A[A
    
     57%|█████▋    | 715/1261 [01:13<01:05,  8.35it/s][A[A
    
     57%|█████▋    | 716/1261 [01:13<01:03,  8.60it/s][A[A
    
     57%|█████▋    | 717/1261 [01:13<01:02,  8.76it/s][A[A
    
     57%|█████▋    | 718/1261 [01:13<01:01,  8.79it/s][A[A
    
     57%|█████▋    | 719/1261 [01:13<01:00,  8.93it/s][A[A
    
     57%|█████▋    | 720/1261 [01:14<01:00,  8.96it/s][A[A
    
     57%|█████▋    | 721/1261 [01:14<01:00,  8.96it/s][A[A
    
     57%|█████▋    | 722/1261 [01:14<00:59,  9.07it/s][A[A
    
     57%|█████▋    | 723/1261 [01:14<00:58,  9.25it/s][A[A
    
     57%|█████▋    | 724/1261 [01:14<00:57,  9.29it/s][A[A
    
     57%|█████▋    | 725/1261 [01:14<00:58,  9.11it/s][A[A
    
     58%|█████▊    | 726/1261 [01:14<00:59,  9.01it/s][A[A
    
     58%|█████▊    | 727/1261 [01:14<01:00,  8.86it/s][A[A
    
     58%|█████▊    | 728/1261 [01:14<00:59,  8.90it/s][A[A
    
     58%|█████▊    | 729/1261 [01:15<00:58,  9.10it/s][A[A
    
     58%|█████▊    | 730/1261 [01:15<00:59,  8.95it/s][A[A
    
     58%|█████▊    | 731/1261 [01:15<00:58,  9.11it/s][A[A
    
     58%|█████▊    | 732/1261 [01:15<00:56,  9.31it/s][A[A
    
     58%|█████▊    | 733/1261 [01:15<00:56,  9.41it/s][A[A
    
     58%|█████▊    | 734/1261 [01:15<00:55,  9.49it/s][A[A
    
     58%|█████▊    | 735/1261 [01:15<00:55,  9.42it/s][A[A
    
     58%|█████▊    | 736/1261 [01:15<00:55,  9.42it/s][A[A
    
     58%|█████▊    | 737/1261 [01:15<00:55,  9.52it/s][A[A
    
     59%|█████▊    | 738/1261 [01:16<00:54,  9.65it/s][A[A
    
     59%|█████▊    | 739/1261 [01:16<00:53,  9.67it/s][A[A
    
     59%|█████▊    | 740/1261 [01:16<00:54,  9.61it/s][A[A
    
     59%|█████▉    | 741/1261 [01:16<00:53,  9.64it/s][A[A
    
     59%|█████▉    | 742/1261 [01:16<00:55,  9.40it/s][A[A
    
     59%|█████▉    | 743/1261 [01:16<00:55,  9.30it/s][A[A
    
     59%|█████▉    | 744/1261 [01:16<00:54,  9.41it/s][A[A
    
     59%|█████▉    | 745/1261 [01:16<00:55,  9.31it/s][A[A
    
     59%|█████▉    | 746/1261 [01:16<00:57,  9.02it/s][A[A
    
     59%|█████▉    | 747/1261 [01:16<00:56,  9.18it/s][A[A
    
     59%|█████▉    | 749/1261 [01:17<00:56,  9.07it/s][A[A
    
     59%|█████▉    | 750/1261 [01:17<00:56,  8.99it/s][A[A
    
     60%|█████▉    | 751/1261 [01:17<00:55,  9.13it/s][A[A
    
     60%|█████▉    | 753/1261 [01:17<00:54,  9.39it/s][A[A
    
     60%|█████▉    | 754/1261 [01:17<00:54,  9.23it/s][A[A
    
     60%|█████▉    | 755/1261 [01:17<00:54,  9.28it/s][A[A
    
     60%|█████▉    | 756/1261 [01:17<00:53,  9.38it/s][A[A
    
     60%|██████    | 757/1261 [01:18<00:53,  9.42it/s][A[A
    
     60%|██████    | 758/1261 [01:18<00:53,  9.48it/s][A[A
    
     60%|██████    | 760/1261 [01:18<00:52,  9.55it/s][A[A
    
     60%|██████    | 761/1261 [01:18<00:53,  9.32it/s][A[A
    
     60%|██████    | 762/1261 [01:18<00:55,  9.00it/s][A[A
    
     61%|██████    | 763/1261 [01:18<00:54,  9.11it/s][A[A
    
     61%|██████    | 764/1261 [01:18<00:54,  9.04it/s][A[A
    
     61%|██████    | 765/1261 [01:18<00:54,  9.02it/s][A[A
    
     61%|██████    | 766/1261 [01:19<00:53,  9.27it/s][A[A
    
     61%|██████    | 767/1261 [01:19<00:52,  9.40it/s][A[A
    
     61%|██████    | 768/1261 [01:19<00:51,  9.53it/s][A[A
    
     61%|██████    | 769/1261 [01:19<00:51,  9.59it/s][A[A
    
     61%|██████    | 770/1261 [01:19<00:52,  9.30it/s][A[A
    
     61%|██████    | 771/1261 [01:19<00:54,  9.00it/s][A[A
    
     61%|██████    | 772/1261 [01:19<00:54,  8.89it/s][A[A
    
     61%|██████▏   | 773/1261 [01:19<00:55,  8.73it/s][A[A
    
     61%|██████▏   | 774/1261 [01:19<00:55,  8.76it/s][A[A
    
     61%|██████▏   | 775/1261 [01:20<00:54,  8.98it/s][A[A
    
     62%|██████▏   | 776/1261 [01:20<00:52,  9.19it/s][A[A
    
     62%|██████▏   | 777/1261 [01:20<00:53,  9.12it/s][A[A
    
     62%|██████▏   | 778/1261 [01:20<00:51,  9.31it/s][A[A
    
     62%|██████▏   | 779/1261 [01:20<00:52,  9.25it/s][A[A
    
     62%|██████▏   | 780/1261 [01:20<00:51,  9.36it/s][A[A
    
     62%|██████▏   | 781/1261 [01:20<00:52,  9.23it/s][A[A
    
     62%|██████▏   | 782/1261 [01:20<00:52,  9.14it/s][A[A
    
     62%|██████▏   | 783/1261 [01:20<00:52,  9.15it/s][A[A
    
     62%|██████▏   | 784/1261 [01:21<00:52,  9.02it/s][A[A
    
     62%|██████▏   | 785/1261 [01:21<00:52,  9.12it/s][A[A
    
     62%|██████▏   | 786/1261 [01:21<00:51,  9.27it/s][A[A
    
     62%|██████▏   | 787/1261 [01:21<00:51,  9.25it/s][A[A
    
     62%|██████▏   | 788/1261 [01:21<00:51,  9.17it/s][A[A
    
     63%|██████▎   | 789/1261 [01:21<00:51,  9.20it/s][A[A
    
     63%|██████▎   | 790/1261 [01:21<00:50,  9.32it/s][A[A
    
     63%|██████▎   | 791/1261 [01:21<00:51,  9.20it/s][A[A
    
     63%|██████▎   | 792/1261 [01:21<00:50,  9.23it/s][A[A
    
     63%|██████▎   | 793/1261 [01:21<00:50,  9.25it/s][A[A
    
     63%|██████▎   | 794/1261 [01:22<00:49,  9.39it/s][A[A
    
     63%|██████▎   | 795/1261 [01:22<00:49,  9.46it/s][A[A
    
     63%|██████▎   | 796/1261 [01:22<00:49,  9.38it/s][A[A
    
     63%|██████▎   | 797/1261 [01:22<00:50,  9.27it/s][A[A
    
     63%|██████▎   | 798/1261 [01:22<00:50,  9.22it/s][A[A
    
     63%|██████▎   | 799/1261 [01:22<00:49,  9.40it/s][A[A
    
     63%|██████▎   | 800/1261 [01:22<00:49,  9.37it/s][A[A
    
     64%|██████▎   | 801/1261 [01:22<00:48,  9.39it/s][A[A
    
     64%|██████▎   | 802/1261 [01:22<00:48,  9.45it/s][A[A
    
     64%|██████▎   | 803/1261 [01:23<00:48,  9.54it/s][A[A
    
     64%|██████▍   | 804/1261 [01:23<00:47,  9.54it/s][A[A
    
     64%|██████▍   | 805/1261 [01:23<00:47,  9.63it/s][A[A
    
     64%|██████▍   | 806/1261 [01:23<00:46,  9.71it/s][A[A
    
     64%|██████▍   | 807/1261 [01:23<00:46,  9.68it/s][A[A
    
     64%|██████▍   | 809/1261 [01:23<00:45,  9.85it/s][A[A
    
     64%|██████▍   | 810/1261 [01:23<00:46,  9.73it/s][A[A
    
     64%|██████▍   | 812/1261 [01:23<00:45,  9.85it/s][A[A
    
     64%|██████▍   | 813/1261 [01:24<00:46,  9.72it/s][A[A
    
     65%|██████▍   | 814/1261 [01:24<00:45,  9.78it/s][A[A
    
     65%|██████▍   | 815/1261 [01:24<00:45,  9.75it/s][A[A
    
     65%|██████▍   | 817/1261 [01:24<00:44,  9.87it/s][A[A
    
     65%|██████▍   | 818/1261 [01:24<00:45,  9.82it/s][A[A
    
     65%|██████▌   | 820/1261 [01:24<00:44,  9.90it/s][A[A
    
     65%|██████▌   | 821/1261 [01:24<00:44,  9.90it/s][A[A
    
     65%|██████▌   | 823/1261 [01:25<00:43,  9.98it/s][A[A
    
     65%|██████▌   | 824/1261 [01:25<00:44,  9.90it/s][A[A
    
     66%|██████▌   | 826/1261 [01:25<00:43,  9.99it/s][A[A
    
     66%|██████▌   | 827/1261 [01:25<00:43,  9.93it/s][A[A
    
     66%|██████▌   | 828/1261 [01:25<00:43,  9.90it/s][A[A
    
     66%|██████▌   | 829/1261 [01:25<00:43,  9.83it/s][A[A
    
     66%|██████▌   | 830/1261 [01:25<00:44,  9.76it/s][A[A
    
     66%|██████▌   | 831/1261 [01:25<00:44,  9.67it/s][A[A
    
     66%|██████▌   | 832/1261 [01:25<00:43,  9.76it/s][A[A
    
     66%|██████▌   | 833/1261 [01:26<00:44,  9.67it/s][A[A
    
     66%|██████▌   | 834/1261 [01:26<00:44,  9.51it/s][A[A
    
     66%|██████▌   | 835/1261 [01:26<00:44,  9.50it/s][A[A
    
     66%|██████▋   | 836/1261 [01:26<00:44,  9.56it/s][A[A
    
     66%|██████▋   | 837/1261 [01:26<00:45,  9.37it/s][A[A
    
     66%|██████▋   | 838/1261 [01:26<00:45,  9.34it/s][A[A
    
     67%|██████▋   | 839/1261 [01:26<00:46,  9.17it/s][A[A
    
     67%|██████▋   | 840/1261 [01:26<00:46,  9.05it/s][A[A
    
     67%|██████▋   | 841/1261 [01:26<00:47,  8.79it/s][A[A
    
     67%|██████▋   | 842/1261 [01:27<00:47,  8.78it/s][A[A
    
     67%|██████▋   | 843/1261 [01:27<00:47,  8.84it/s][A[A
    
     67%|██████▋   | 844/1261 [01:27<00:48,  8.61it/s][A[A
    
     67%|██████▋   | 845/1261 [01:27<00:48,  8.61it/s][A[A
    
     67%|██████▋   | 846/1261 [01:27<00:47,  8.70it/s][A[A
    
     67%|██████▋   | 847/1261 [01:27<00:46,  8.89it/s][A[A
    
     67%|██████▋   | 848/1261 [01:27<00:46,  8.89it/s][A[A
    
     67%|██████▋   | 849/1261 [01:27<00:45,  8.98it/s][A[A
    
     67%|██████▋   | 850/1261 [01:27<00:45,  8.95it/s][A[A
    
     67%|██████▋   | 851/1261 [01:28<00:45,  9.11it/s][A[A
    
     68%|██████▊   | 852/1261 [01:28<00:44,  9.17it/s][A[A
    
     68%|██████▊   | 853/1261 [01:28<00:44,  9.17it/s][A[A
    
     68%|██████▊   | 854/1261 [01:28<00:44,  9.16it/s][A[A
    
     68%|██████▊   | 855/1261 [01:28<00:43,  9.27it/s][A[A
    
     68%|██████▊   | 857/1261 [01:28<00:42,  9.46it/s][A[A
    
     68%|██████▊   | 858/1261 [01:28<00:42,  9.55it/s][A[A
    
     68%|██████▊   | 859/1261 [01:28<00:42,  9.57it/s][A[A
    
     68%|██████▊   | 860/1261 [01:29<00:41,  9.61it/s][A[A
    
     68%|██████▊   | 861/1261 [01:29<00:41,  9.63it/s][A[A
    
     68%|██████▊   | 862/1261 [01:29<00:41,  9.54it/s][A[A
    
     68%|██████▊   | 863/1261 [01:29<00:42,  9.45it/s][A[A
    
     69%|██████▊   | 864/1261 [01:29<00:42,  9.40it/s][A[A
    
     69%|██████▊   | 865/1261 [01:29<00:42,  9.29it/s][A[A
    
     69%|██████▊   | 866/1261 [01:29<00:42,  9.34it/s][A[A
    
     69%|██████▉   | 867/1261 [01:29<00:43,  9.15it/s][A[A
    
     69%|██████▉   | 868/1261 [01:29<00:42,  9.23it/s][A[A
    
     69%|██████▉   | 869/1261 [01:29<00:42,  9.17it/s][A[A
    
     69%|██████▉   | 870/1261 [01:30<00:42,  9.19it/s][A[A
    
     69%|██████▉   | 871/1261 [01:30<00:42,  9.13it/s][A[A
    
     69%|██████▉   | 873/1261 [01:30<00:41,  9.37it/s][A[A
    
     69%|██████▉   | 874/1261 [01:30<00:41,  9.42it/s][A[A
    
     69%|██████▉   | 875/1261 [01:30<00:40,  9.44it/s][A[A
    
     69%|██████▉   | 876/1261 [01:30<00:41,  9.35it/s][A[A
    
     70%|██████▉   | 877/1261 [01:30<00:41,  9.26it/s][A[A
    
     70%|██████▉   | 878/1261 [01:30<00:42,  9.08it/s][A[A
    
     70%|██████▉   | 879/1261 [01:31<00:41,  9.15it/s][A[A
    
     70%|██████▉   | 880/1261 [01:31<00:41,  9.13it/s][A[A
    
     70%|██████▉   | 881/1261 [01:31<00:41,  9.20it/s][A[A
    
     70%|██████▉   | 882/1261 [01:31<00:41,  9.20it/s][A[A
    
     70%|███████   | 883/1261 [01:31<00:40,  9.41it/s][A[A
    
     70%|███████   | 884/1261 [01:31<00:39,  9.55it/s][A[A
    
     70%|███████   | 885/1261 [01:31<00:39,  9.61it/s][A[A
    
     70%|███████   | 886/1261 [01:31<00:39,  9.61it/s][A[A
    
     70%|███████   | 887/1261 [01:31<00:38,  9.69it/s][A[A
    
     70%|███████   | 888/1261 [01:32<00:39,  9.54it/s][A[A
    
     70%|███████   | 889/1261 [01:32<00:38,  9.56it/s][A[A
    
     71%|███████   | 890/1261 [01:32<00:39,  9.41it/s][A[A
    
     71%|███████   | 891/1261 [01:32<00:39,  9.37it/s][A[A
    
     71%|███████   | 892/1261 [01:32<00:39,  9.33it/s][A[A
    
     71%|███████   | 893/1261 [01:32<00:39,  9.21it/s][A[A
    
     71%|███████   | 894/1261 [01:32<00:40,  9.15it/s][A[A
    
     71%|███████   | 895/1261 [01:32<00:40,  8.94it/s][A[A
    
     71%|███████   | 896/1261 [01:32<00:39,  9.13it/s][A[A
    
     71%|███████   | 897/1261 [01:32<00:39,  9.32it/s][A[A
    
     71%|███████   | 898/1261 [01:33<00:38,  9.51it/s][A[A
    
     71%|███████▏  | 899/1261 [01:33<00:38,  9.50it/s][A[A
    
     71%|███████▏  | 900/1261 [01:33<00:37,  9.54it/s][A[A
    
     71%|███████▏  | 901/1261 [01:33<00:37,  9.58it/s][A[A
    
     72%|███████▏  | 902/1261 [01:33<00:37,  9.55it/s][A[A
    
     72%|███████▏  | 903/1261 [01:33<00:37,  9.44it/s][A[A
    
     72%|███████▏  | 904/1261 [01:33<00:38,  9.24it/s][A[A
    
     72%|███████▏  | 905/1261 [01:33<00:38,  9.30it/s][A[A
    
     72%|███████▏  | 906/1261 [01:33<00:37,  9.37it/s][A[A
    
     72%|███████▏  | 907/1261 [01:34<00:38,  9.24it/s][A[A
    
     72%|███████▏  | 908/1261 [01:34<00:38,  9.21it/s][A[A
    
     72%|███████▏  | 909/1261 [01:34<00:38,  9.11it/s][A[A
    
     72%|███████▏  | 910/1261 [01:34<00:38,  9.21it/s][A[A
    
     72%|███████▏  | 911/1261 [01:34<00:37,  9.23it/s][A[A
    
     72%|███████▏  | 912/1261 [01:34<00:37,  9.22it/s][A[A
    
     72%|███████▏  | 913/1261 [01:34<00:37,  9.33it/s][A[A
    
     72%|███████▏  | 914/1261 [01:34<00:37,  9.35it/s][A[A
    
     73%|███████▎  | 915/1261 [01:34<00:37,  9.20it/s][A[A
    
     73%|███████▎  | 916/1261 [01:35<00:37,  9.12it/s][A[A
    
     73%|███████▎  | 917/1261 [01:35<00:37,  9.16it/s][A[A
    
     73%|███████▎  | 918/1261 [01:35<00:37,  9.06it/s][A[A
    
     73%|███████▎  | 919/1261 [01:35<00:37,  9.09it/s][A[A
    
     73%|███████▎  | 920/1261 [01:35<00:37,  9.06it/s][A[A
    
     73%|███████▎  | 921/1261 [01:35<00:36,  9.22it/s][A[A
    
     73%|███████▎  | 922/1261 [01:35<00:36,  9.21it/s][A[A
    
     73%|███████▎  | 923/1261 [01:35<00:36,  9.34it/s][A[A
    
     73%|███████▎  | 924/1261 [01:35<00:35,  9.41it/s][A[A
    
     73%|███████▎  | 925/1261 [01:36<00:36,  9.30it/s][A[A
    
     73%|███████▎  | 926/1261 [01:36<00:36,  9.22it/s][A[A
    
     74%|███████▎  | 927/1261 [01:36<00:36,  9.11it/s][A[A
    
     74%|███████▎  | 928/1261 [01:36<00:36,  9.10it/s][A[A
    
     74%|███████▎  | 929/1261 [01:36<00:37,  8.80it/s][A[A
    
     74%|███████▍  | 930/1261 [01:36<00:37,  8.74it/s][A[A
    
     74%|███████▍  | 931/1261 [01:36<00:37,  8.76it/s][A[A
    
     74%|███████▍  | 932/1261 [01:36<00:37,  8.86it/s][A[A
    
     74%|███████▍  | 933/1261 [01:36<00:36,  8.95it/s][A[A
    
     74%|███████▍  | 934/1261 [01:37<00:35,  9.10it/s][A[A
    
     74%|███████▍  | 935/1261 [01:37<00:35,  9.12it/s][A[A
    
     74%|███████▍  | 936/1261 [01:37<00:35,  9.20it/s][A[A
    
     74%|███████▍  | 937/1261 [01:37<00:35,  9.25it/s][A[A
    
     74%|███████▍  | 938/1261 [01:37<00:34,  9.42it/s][A[A
    
     74%|███████▍  | 939/1261 [01:37<00:34,  9.35it/s][A[A
    
     75%|███████▍  | 940/1261 [01:37<00:34,  9.33it/s][A[A
    
     75%|███████▍  | 941/1261 [01:37<00:35,  9.08it/s][A[A
    
     75%|███████▍  | 942/1261 [01:37<00:35,  9.04it/s][A[A
    
     75%|███████▍  | 943/1261 [01:37<00:35,  9.06it/s][A[A
    
     75%|███████▍  | 944/1261 [01:38<00:34,  9.10it/s][A[A
    
     75%|███████▍  | 945/1261 [01:38<00:35,  9.00it/s][A[A
    
     75%|███████▌  | 946/1261 [01:38<00:34,  9.10it/s][A[A
    
     75%|███████▌  | 947/1261 [01:38<00:34,  9.18it/s][A[A
    
     75%|███████▌  | 948/1261 [01:38<00:33,  9.28it/s][A[A
    
     75%|███████▌  | 949/1261 [01:38<00:33,  9.38it/s][A[A
    
     75%|███████▌  | 950/1261 [01:38<00:33,  9.42it/s][A[A
    
     75%|███████▌  | 951/1261 [01:38<00:33,  9.33it/s][A[A
    
     75%|███████▌  | 952/1261 [01:38<00:32,  9.44it/s][A[A
    
     76%|███████▌  | 953/1261 [01:39<00:32,  9.34it/s][A[A
    
     76%|███████▌  | 954/1261 [01:39<00:33,  9.19it/s][A[A
    
     76%|███████▌  | 955/1261 [01:39<00:34,  8.85it/s][A[A
    
     76%|███████▌  | 956/1261 [01:39<00:35,  8.49it/s][A[A
    
     76%|███████▌  | 957/1261 [01:39<00:38,  7.89it/s][A[A
    
     76%|███████▌  | 958/1261 [01:39<00:38,  7.78it/s][A[A
    
     76%|███████▌  | 959/1261 [01:39<00:39,  7.56it/s][A[A
    
     76%|███████▌  | 960/1261 [01:39<00:38,  7.89it/s][A[A
    
     76%|███████▌  | 961/1261 [01:40<00:37,  8.03it/s][A[A
    
     76%|███████▋  | 962/1261 [01:40<00:36,  8.27it/s][A[A
    
     76%|███████▋  | 963/1261 [01:40<00:36,  8.26it/s][A[A
    
     76%|███████▋  | 964/1261 [01:40<00:35,  8.42it/s][A[A
    
     77%|███████▋  | 965/1261 [01:40<00:34,  8.53it/s][A[A
    
     77%|███████▋  | 966/1261 [01:40<00:34,  8.65it/s][A[A
    
     77%|███████▋  | 967/1261 [01:40<00:34,  8.59it/s][A[A
    
     77%|███████▋  | 968/1261 [01:40<00:34,  8.58it/s][A[A
    
     77%|███████▋  | 969/1261 [01:41<00:35,  8.34it/s][A[A
    
     77%|███████▋  | 970/1261 [01:41<00:34,  8.44it/s][A[A
    
     77%|███████▋  | 971/1261 [01:41<00:35,  8.27it/s][A[A
    
     77%|███████▋  | 972/1261 [01:41<00:34,  8.29it/s][A[A
    
     77%|███████▋  | 973/1261 [01:41<00:34,  8.39it/s][A[A
    
     77%|███████▋  | 974/1261 [01:41<00:33,  8.45it/s][A[A
    
     77%|███████▋  | 975/1261 [01:41<00:33,  8.42it/s][A[A
    
     77%|███████▋  | 976/1261 [01:41<00:34,  8.35it/s][A[A
    
     77%|███████▋  | 977/1261 [01:41<00:33,  8.55it/s][A[A
    
     78%|███████▊  | 978/1261 [01:42<00:32,  8.64it/s][A[A
    
     78%|███████▊  | 979/1261 [01:42<00:31,  8.84it/s][A[A
    
     78%|███████▊  | 980/1261 [01:42<00:31,  8.96it/s][A[A
    
     78%|███████▊  | 981/1261 [01:42<00:31,  9.01it/s][A[A
    
     78%|███████▊  | 982/1261 [01:42<00:31,  8.95it/s][A[A
    
     78%|███████▊  | 983/1261 [01:42<00:30,  9.12it/s][A[A
    
     78%|███████▊  | 984/1261 [01:42<00:30,  9.20it/s][A[A
    
     78%|███████▊  | 985/1261 [01:42<00:29,  9.30it/s][A[A
    
     78%|███████▊  | 987/1261 [01:43<00:28,  9.57it/s][A[A
    
     78%|███████▊  | 988/1261 [01:43<00:28,  9.63it/s][A[A
    
     78%|███████▊  | 989/1261 [01:43<00:28,  9.60it/s][A[A
    
     79%|███████▊  | 990/1261 [01:43<00:29,  9.24it/s][A[A
    
     79%|███████▊  | 991/1261 [01:43<00:29,  9.11it/s][A[A
    
     79%|███████▊  | 992/1261 [01:43<00:29,  9.01it/s][A[A
    
     79%|███████▊  | 993/1261 [01:43<00:30,  8.82it/s][A[A
    
     79%|███████▉  | 994/1261 [01:43<00:30,  8.63it/s][A[A
    
     79%|███████▉  | 995/1261 [01:43<00:30,  8.61it/s][A[A
    
     79%|███████▉  | 996/1261 [01:44<00:30,  8.73it/s][A[A
    
     79%|███████▉  | 997/1261 [01:44<00:30,  8.78it/s][A[A
    
     79%|███████▉  | 998/1261 [01:44<00:29,  8.89it/s][A[A
    
     79%|███████▉  | 999/1261 [01:44<00:30,  8.73it/s][A[A
    
     79%|███████▉  | 1000/1261 [01:44<00:29,  8.92it/s][A[A
    
     79%|███████▉  | 1001/1261 [01:44<00:28,  9.10it/s][A[A
    
     79%|███████▉  | 1002/1261 [01:44<00:28,  9.18it/s][A[A
    
     80%|███████▉  | 1004/1261 [01:44<00:26,  9.63it/s][A[A
    
     80%|███████▉  | 1006/1261 [01:45<00:25,  9.82it/s][A[A
    
     80%|███████▉  | 1007/1261 [01:45<00:27,  9.35it/s][A[A
    
     80%|███████▉  | 1008/1261 [01:45<00:27,  9.14it/s][A[A
    
     80%|████████  | 1009/1261 [01:45<00:27,  9.16it/s][A[A
    
     80%|████████  | 1011/1261 [01:45<00:26,  9.43it/s][A[A
    
     80%|████████  | 1012/1261 [01:45<00:26,  9.23it/s][A[A
    
     80%|████████  | 1014/1261 [01:45<00:25,  9.61it/s][A[A
    
     81%|████████  | 1016/1261 [01:46<00:25,  9.74it/s][A[A
    
     81%|████████  | 1017/1261 [01:46<00:25,  9.61it/s][A[A
    
     81%|████████  | 1018/1261 [01:46<00:25,  9.44it/s][A[A
    
     81%|████████  | 1019/1261 [01:46<00:26,  9.30it/s][A[A
    
     81%|████████  | 1020/1261 [01:46<00:26,  9.24it/s][A[A
    
     81%|████████  | 1021/1261 [01:46<00:25,  9.34it/s][A[A
    
     81%|████████  | 1022/1261 [01:46<00:26,  9.19it/s][A[A
    
     81%|████████  | 1023/1261 [01:46<00:25,  9.22it/s][A[A
    
     81%|████████  | 1024/1261 [01:47<00:26,  9.00it/s][A[A
    
     81%|████████▏ | 1025/1261 [01:47<00:25,  9.19it/s][A[A
    
     81%|████████▏ | 1026/1261 [01:47<00:24,  9.41it/s][A[A
    
     81%|████████▏ | 1027/1261 [01:47<00:24,  9.45it/s][A[A
    
     82%|████████▏ | 1028/1261 [01:47<00:24,  9.51it/s][A[A
    
     82%|████████▏ | 1029/1261 [01:47<00:24,  9.53it/s][A[A
    
     82%|████████▏ | 1030/1261 [01:47<00:24,  9.42it/s][A[A
    
     82%|████████▏ | 1031/1261 [01:47<00:24,  9.49it/s][A[A
    
     82%|████████▏ | 1032/1261 [01:47<00:24,  9.40it/s][A[A
    
     82%|████████▏ | 1033/1261 [01:47<00:24,  9.49it/s][A[A
    
     82%|████████▏ | 1034/1261 [01:48<00:24,  9.35it/s][A[A
    
     82%|████████▏ | 1035/1261 [01:48<00:24,  9.40it/s][A[A
    
     82%|████████▏ | 1036/1261 [01:48<00:24,  9.29it/s][A[A
    
     82%|████████▏ | 1037/1261 [01:48<00:24,  9.22it/s][A[A
    
     82%|████████▏ | 1038/1261 [01:48<00:24,  9.13it/s][A[A
    
     82%|████████▏ | 1039/1261 [01:48<00:25,  8.75it/s][A[A
    
     82%|████████▏ | 1040/1261 [01:48<00:25,  8.68it/s][A[A
    
     83%|████████▎ | 1041/1261 [01:48<00:25,  8.70it/s][A[A
    
     83%|████████▎ | 1042/1261 [01:48<00:24,  8.90it/s][A[A
    
     83%|████████▎ | 1043/1261 [01:49<00:24,  8.92it/s][A[A
    
     83%|████████▎ | 1044/1261 [01:49<00:24,  8.94it/s][A[A
    
     83%|████████▎ | 1045/1261 [01:49<00:24,  8.77it/s][A[A
    
     83%|████████▎ | 1046/1261 [01:49<00:24,  8.70it/s][A[A
    
     83%|████████▎ | 1047/1261 [01:49<00:24,  8.61it/s][A[A
    
     83%|████████▎ | 1048/1261 [01:49<00:23,  8.97it/s][A[A
    
     83%|████████▎ | 1049/1261 [01:49<00:22,  9.23it/s][A[A
    
     83%|████████▎ | 1050/1261 [01:49<00:23,  9.02it/s][A[A
    
     83%|████████▎ | 1051/1261 [01:49<00:23,  8.97it/s][A[A
    
     83%|████████▎ | 1052/1261 [01:50<00:23,  9.06it/s][A[A
    
     84%|████████▎ | 1053/1261 [01:50<00:22,  9.25it/s][A[A
    
     84%|████████▎ | 1054/1261 [01:50<00:22,  9.28it/s][A[A
    
     84%|████████▎ | 1055/1261 [01:50<00:22,  9.30it/s][A[A
    
     84%|████████▎ | 1056/1261 [01:50<00:21,  9.33it/s][A[A
    
     84%|████████▍ | 1057/1261 [01:50<00:22,  9.12it/s][A[A
    
     84%|████████▍ | 1058/1261 [01:50<00:22,  8.98it/s][A[A
    
     84%|████████▍ | 1059/1261 [01:50<00:22,  9.10it/s][A[A
    
     84%|████████▍ | 1060/1261 [01:50<00:21,  9.31it/s][A[A
    
     84%|████████▍ | 1062/1261 [01:51<00:21,  9.48it/s][A[A
    
     84%|████████▍ | 1063/1261 [01:51<00:21,  9.28it/s][A[A
    
     84%|████████▍ | 1064/1261 [01:51<00:21,  9.19it/s][A[A
    
     84%|████████▍ | 1065/1261 [01:51<00:21,  9.06it/s][A[A
    
     85%|████████▍ | 1066/1261 [01:51<00:21,  8.93it/s][A[A
    
     85%|████████▍ | 1067/1261 [01:51<00:22,  8.53it/s][A[A
    
     85%|████████▍ | 1068/1261 [01:51<00:23,  8.34it/s][A[A
    
     85%|████████▍ | 1069/1261 [01:51<00:22,  8.60it/s][A[A
    
     85%|████████▍ | 1070/1261 [01:52<00:21,  8.82it/s][A[A
    
     85%|████████▍ | 1071/1261 [01:52<00:21,  8.99it/s][A[A
    
     85%|████████▌ | 1072/1261 [01:52<00:21,  8.92it/s][A[A
    
     85%|████████▌ | 1073/1261 [01:52<00:21,  8.79it/s][A[A
    
     85%|████████▌ | 1074/1261 [01:52<00:21,  8.74it/s][A[A
    
     85%|████████▌ | 1075/1261 [01:52<00:21,  8.70it/s][A[A
    
     85%|████████▌ | 1076/1261 [01:52<00:22,  8.38it/s][A[A
    
     85%|████████▌ | 1077/1261 [01:52<00:21,  8.48it/s][A[A
    
     85%|████████▌ | 1078/1261 [01:52<00:21,  8.57it/s][A[A
    
     86%|████████▌ | 1079/1261 [01:53<00:21,  8.55it/s][A[A
    
     86%|████████▌ | 1080/1261 [01:53<00:21,  8.53it/s][A[A
    
     86%|████████▌ | 1081/1261 [01:53<00:21,  8.43it/s][A[A
    
     86%|████████▌ | 1082/1261 [01:53<00:21,  8.35it/s][A[A
    
     86%|████████▌ | 1083/1261 [01:53<00:21,  8.44it/s][A[A
    
     86%|████████▌ | 1084/1261 [01:53<00:20,  8.50it/s][A[A
    
     86%|████████▌ | 1085/1261 [01:53<00:20,  8.67it/s][A[A
    
     86%|████████▌ | 1086/1261 [01:53<00:19,  8.83it/s][A[A
    
     86%|████████▌ | 1087/1261 [01:54<00:19,  8.94it/s][A[A
    
     86%|████████▋ | 1088/1261 [01:54<00:19,  9.02it/s][A[A
    
     86%|████████▋ | 1089/1261 [01:54<00:19,  9.05it/s][A[A
    
     86%|████████▋ | 1090/1261 [01:54<00:18,  9.01it/s][A[A
    
     87%|████████▋ | 1091/1261 [01:54<00:18,  9.14it/s][A[A
    
     87%|████████▋ | 1092/1261 [01:54<00:18,  9.27it/s][A[A
    
     87%|████████▋ | 1093/1261 [01:54<00:17,  9.43it/s][A[A
    
     87%|████████▋ | 1094/1261 [01:54<00:17,  9.29it/s][A[A
    
     87%|████████▋ | 1095/1261 [01:54<00:17,  9.32it/s][A[A
    
     87%|████████▋ | 1096/1261 [01:54<00:17,  9.50it/s][A[A
    
     87%|████████▋ | 1097/1261 [01:55<00:17,  9.54it/s][A[A
    
     87%|████████▋ | 1098/1261 [01:55<00:17,  9.54it/s][A[A
    
     87%|████████▋ | 1099/1261 [01:55<00:16,  9.61it/s][A[A
    
     87%|████████▋ | 1100/1261 [01:55<00:16,  9.71it/s][A[A
    
     87%|████████▋ | 1101/1261 [01:55<00:16,  9.74it/s][A[A
    
     87%|████████▋ | 1102/1261 [01:55<00:16,  9.53it/s][A[A
    
     87%|████████▋ | 1103/1261 [01:55<00:16,  9.53it/s][A[A
    
     88%|████████▊ | 1104/1261 [01:55<00:16,  9.61it/s][A[A
    
     88%|████████▊ | 1105/1261 [01:55<00:16,  9.61it/s][A[A
    
     88%|████████▊ | 1106/1261 [01:56<00:16,  9.65it/s][A[A
    
     88%|████████▊ | 1107/1261 [01:56<00:16,  9.57it/s][A[A
    
     88%|████████▊ | 1108/1261 [01:56<00:16,  9.11it/s][A[A
    
     88%|████████▊ | 1109/1261 [01:56<00:16,  9.04it/s][A[A
    
     88%|████████▊ | 1110/1261 [01:56<00:16,  9.07it/s][A[A
    
     88%|████████▊ | 1111/1261 [01:56<00:16,  9.20it/s][A[A
    
     88%|████████▊ | 1112/1261 [01:56<00:15,  9.38it/s][A[A
    
     88%|████████▊ | 1113/1261 [01:56<00:15,  9.46it/s][A[A
    
     88%|████████▊ | 1114/1261 [01:56<00:15,  9.40it/s][A[A
    
     88%|████████▊ | 1115/1261 [01:57<00:15,  9.40it/s][A[A
    
     89%|████████▊ | 1116/1261 [01:57<00:15,  9.21it/s][A[A
    
     89%|████████▊ | 1117/1261 [01:57<00:15,  9.28it/s][A[A
    
     89%|████████▊ | 1118/1261 [01:57<00:15,  9.23it/s][A[A
    
     89%|████████▊ | 1119/1261 [01:57<00:15,  9.33it/s][A[A
    
     89%|████████▉ | 1120/1261 [01:57<00:14,  9.49it/s][A[A
    
     89%|████████▉ | 1121/1261 [01:57<00:14,  9.50it/s][A[A
    
     89%|████████▉ | 1122/1261 [01:57<00:14,  9.58it/s][A[A
    
     89%|████████▉ | 1123/1261 [01:57<00:14,  9.60it/s][A[A
    
     89%|████████▉ | 1124/1261 [01:57<00:14,  9.69it/s][A[A
    
     89%|████████▉ | 1126/1261 [01:58<00:13,  9.73it/s][A[A
    
     89%|████████▉ | 1127/1261 [01:58<00:13,  9.81it/s][A[A
    
     89%|████████▉ | 1128/1261 [01:58<00:13,  9.76it/s][A[A
    
     90%|████████▉ | 1130/1261 [01:58<00:13,  9.79it/s][A[A
    
     90%|████████▉ | 1131/1261 [01:58<00:13,  9.83it/s][A[A
    
     90%|████████▉ | 1132/1261 [01:58<00:13,  9.78it/s][A[A
    
     90%|████████▉ | 1133/1261 [01:58<00:13,  9.83it/s][A[A
    
     90%|████████▉ | 1134/1261 [01:58<00:13,  9.69it/s][A[A
    
     90%|█████████ | 1135/1261 [01:59<00:12,  9.72it/s][A[A
    
     90%|█████████ | 1136/1261 [01:59<00:12,  9.74it/s][A[A
    
     90%|█████████ | 1137/1261 [01:59<00:12,  9.71it/s][A[A
    
     90%|█████████ | 1138/1261 [01:59<00:12,  9.68it/s][A[A
    
     90%|█████████ | 1139/1261 [01:59<00:12,  9.68it/s][A[A
    
     90%|█████████ | 1140/1261 [01:59<00:12,  9.68it/s][A[A
    
     90%|█████████ | 1141/1261 [01:59<00:12,  9.53it/s][A[A
    
     91%|█████████ | 1142/1261 [01:59<00:12,  9.58it/s][A[A
    
     91%|█████████ | 1143/1261 [01:59<00:12,  9.63it/s][A[A
    
     91%|█████████ | 1144/1261 [02:00<00:12,  9.74it/s][A[A
    
     91%|█████████ | 1145/1261 [02:00<00:12,  9.62it/s][A[A
    
     91%|█████████ | 1146/1261 [02:00<00:11,  9.68it/s][A[A
    
     91%|█████████ | 1147/1261 [02:00<00:11,  9.59it/s][A[A
    
     91%|█████████ | 1148/1261 [02:00<00:11,  9.64it/s][A[A
    
     91%|█████████ | 1149/1261 [02:00<00:11,  9.60it/s][A[A
    
     91%|█████████ | 1150/1261 [02:00<00:11,  9.66it/s][A[A
    
     91%|█████████▏| 1151/1261 [02:00<00:11,  9.52it/s][A[A
    
     91%|█████████▏| 1152/1261 [02:00<00:11,  9.53it/s][A[A
    
     91%|█████████▏| 1153/1261 [02:00<00:11,  9.18it/s][A[A
    
     92%|█████████▏| 1154/1261 [02:01<00:12,  8.77it/s][A[A
    
     92%|█████████▏| 1155/1261 [02:01<00:12,  8.78it/s][A[A
    
     92%|█████████▏| 1156/1261 [02:01<00:12,  8.12it/s][A[A
    
     92%|█████████▏| 1157/1261 [02:01<00:13,  7.98it/s][A[A
    
     92%|█████████▏| 1158/1261 [02:01<00:12,  8.13it/s][A[A
    
     92%|█████████▏| 1159/1261 [02:01<00:12,  8.33it/s][A[A
    
     92%|█████████▏| 1160/1261 [02:01<00:12,  8.39it/s][A[A
    
     92%|█████████▏| 1161/1261 [02:01<00:11,  8.59it/s][A[A
    
     92%|█████████▏| 1162/1261 [02:02<00:11,  8.56it/s][A[A
    
     92%|█████████▏| 1163/1261 [02:02<00:11,  8.79it/s][A[A
    
     92%|█████████▏| 1164/1261 [02:02<00:10,  9.09it/s][A[A
    
     92%|█████████▏| 1165/1261 [02:02<00:10,  9.15it/s][A[A
    
     92%|█████████▏| 1166/1261 [02:02<00:10,  9.29it/s][A[A
    
     93%|█████████▎| 1167/1261 [02:02<00:10,  9.36it/s][A[A
    
     93%|█████████▎| 1168/1261 [02:02<00:11,  8.33it/s][A[A
    
     93%|█████████▎| 1169/1261 [02:02<00:11,  8.21it/s][A[A
    
     93%|█████████▎| 1170/1261 [02:02<00:10,  8.60it/s][A[A
    
     93%|█████████▎| 1171/1261 [02:03<00:10,  8.88it/s][A[A
    
     93%|█████████▎| 1172/1261 [02:03<00:09,  9.11it/s][A[A
    
     93%|█████████▎| 1173/1261 [02:03<00:09,  9.17it/s][A[A
    
     93%|█████████▎| 1174/1261 [02:03<00:09,  9.15it/s][A[A
    
     93%|█████████▎| 1175/1261 [02:03<00:09,  9.19it/s][A[A
    
     93%|█████████▎| 1176/1261 [02:03<00:09,  9.24it/s][A[A
    
     93%|█████████▎| 1177/1261 [02:03<00:08,  9.40it/s][A[A
    
     93%|█████████▎| 1178/1261 [02:03<00:08,  9.46it/s][A[A
    
     93%|█████████▎| 1179/1261 [02:03<00:08,  9.38it/s][A[A
    
     94%|█████████▎| 1180/1261 [02:04<00:08,  9.48it/s][A[A
    
     94%|█████████▎| 1181/1261 [02:04<00:08,  9.53it/s][A[A
    
     94%|█████████▎| 1182/1261 [02:04<00:08,  9.52it/s][A[A
    
     94%|█████████▍| 1183/1261 [02:04<00:08,  9.46it/s][A[A
    
     94%|█████████▍| 1184/1261 [02:04<00:08,  8.77it/s][A[A
    
     94%|█████████▍| 1185/1261 [02:04<00:08,  8.78it/s][A[A
    
     94%|█████████▍| 1186/1261 [02:04<00:08,  9.00it/s][A[A
    
     94%|█████████▍| 1187/1261 [02:04<00:08,  9.25it/s][A[A
    
     94%|█████████▍| 1188/1261 [02:04<00:07,  9.32it/s][A[A
    
     94%|█████████▍| 1189/1261 [02:04<00:07,  9.22it/s][A[A
    
     94%|█████████▍| 1190/1261 [02:05<00:07,  9.05it/s][A[A
    
     94%|█████████▍| 1191/1261 [02:05<00:07,  9.09it/s][A[A
    
     95%|█████████▍| 1192/1261 [02:05<00:07,  9.19it/s][A[A
    
     95%|█████████▍| 1194/1261 [02:05<00:07,  9.44it/s][A[A
    
     95%|█████████▍| 1195/1261 [02:05<00:06,  9.53it/s][A[A
    
     95%|█████████▍| 1196/1261 [02:05<00:06,  9.56it/s][A[A
    
     95%|█████████▍| 1197/1261 [02:05<00:06,  9.55it/s][A[A
    
     95%|█████████▌| 1199/1261 [02:06<00:06,  9.71it/s][A[A
    
     95%|█████████▌| 1200/1261 [02:06<00:06,  9.78it/s][A[A
    
     95%|█████████▌| 1201/1261 [02:06<00:06,  9.80it/s][A[A
    
     95%|█████████▌| 1202/1261 [02:06<00:06,  9.80it/s][A[A
    
     95%|█████████▌| 1203/1261 [02:06<00:05,  9.79it/s][A[A
    
     96%|█████████▌| 1205/1261 [02:06<00:05,  9.82it/s][A[A
    
     96%|█████████▌| 1207/1261 [02:06<00:05,  9.87it/s][A[A
    
     96%|█████████▌| 1208/1261 [02:06<00:05,  9.87it/s][A[A
    
     96%|█████████▌| 1209/1261 [02:07<00:05,  9.87it/s][A[A
    
     96%|█████████▌| 1210/1261 [02:07<00:05,  9.82it/s][A[A
    
     96%|█████████▌| 1212/1261 [02:07<00:04,  9.83it/s][A[A
    
     96%|█████████▌| 1213/1261 [02:07<00:04,  9.85it/s][A[A
    
     96%|█████████▋| 1214/1261 [02:07<00:04,  9.84it/s][A[A
    
     96%|█████████▋| 1215/1261 [02:07<00:04,  9.82it/s][A[A
    
     96%|█████████▋| 1216/1261 [02:07<00:04,  9.81it/s][A[A
    
     97%|█████████▋| 1217/1261 [02:07<00:04,  9.77it/s][A[A
    
     97%|█████████▋| 1218/1261 [02:07<00:04,  9.79it/s][A[A
    
     97%|█████████▋| 1219/1261 [02:08<00:04,  9.78it/s][A[A
    
     97%|█████████▋| 1220/1261 [02:08<00:04,  9.78it/s][A[A
    
     97%|█████████▋| 1221/1261 [02:08<00:04,  9.79it/s][A[A
    
     97%|█████████▋| 1222/1261 [02:08<00:03,  9.79it/s][A[A
    
     97%|█████████▋| 1223/1261 [02:08<00:03,  9.80it/s][A[A
    
     97%|█████████▋| 1224/1261 [02:08<00:03,  9.81it/s][A[A
    
     97%|█████████▋| 1225/1261 [02:08<00:03,  9.68it/s][A[A
    
     97%|█████████▋| 1226/1261 [02:08<00:03,  9.69it/s][A[A
    
     97%|█████████▋| 1227/1261 [02:08<00:03,  9.68it/s][A[A
    
     97%|█████████▋| 1228/1261 [02:08<00:03,  9.69it/s][A[A
    
     97%|█████████▋| 1229/1261 [02:09<00:03,  9.64it/s][A[A
    
     98%|█████████▊| 1230/1261 [02:09<00:03,  9.69it/s][A[A
    
     98%|█████████▊| 1231/1261 [02:09<00:03,  9.43it/s][A[A
    
     98%|█████████▊| 1232/1261 [02:09<00:03,  9.40it/s][A[A
    
     98%|█████████▊| 1233/1261 [02:09<00:02,  9.50it/s][A[A
    
     98%|█████████▊| 1234/1261 [02:09<00:02,  9.04it/s][A[A
    
     98%|█████████▊| 1235/1261 [02:09<00:02,  8.79it/s][A[A
    
     98%|█████████▊| 1236/1261 [02:09<00:02,  8.89it/s][A[A
    
     98%|█████████▊| 1237/1261 [02:09<00:02,  8.66it/s][A[A
    
     98%|█████████▊| 1238/1261 [02:10<00:02,  8.96it/s][A[A
    
     98%|█████████▊| 1239/1261 [02:10<00:02,  9.20it/s][A[A
    
     98%|█████████▊| 1241/1261 [02:10<00:02,  9.39it/s][A[A
    
     98%|█████████▊| 1242/1261 [02:10<00:02,  9.45it/s][A[A
    
     99%|█████████▊| 1243/1261 [02:10<00:01,  9.48it/s][A[A
    
     99%|█████████▊| 1244/1261 [02:10<00:01,  9.42it/s][A[A
    
     99%|█████████▊| 1245/1261 [02:10<00:01,  9.35it/s][A[A
    
     99%|█████████▉| 1246/1261 [02:10<00:01,  9.41it/s][A[A
    
     99%|█████████▉| 1247/1261 [02:11<00:01,  9.41it/s][A[A
    
     99%|█████████▉| 1249/1261 [02:11<00:01,  9.55it/s][A[A
    
     99%|█████████▉| 1251/1261 [02:11<00:01,  9.72it/s][A[A
    
     99%|█████████▉| 1253/1261 [02:11<00:00,  9.71it/s][A[A
    
     99%|█████████▉| 1254/1261 [02:11<00:00,  9.71it/s][A[A
    
    100%|█████████▉| 1256/1261 [02:11<00:00,  9.89it/s][A[A
    
    100%|█████████▉| 1257/1261 [02:12<00:00,  9.90it/s][A[A
    
    100%|█████████▉| 1258/1261 [02:12<00:00,  9.92it/s][A[A
    
    100%|█████████▉| 1260/1261 [02:12<00:00,  9.96it/s][A[A
    
    [A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test.mp4 
    



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))
```





<video width="960" height="540" controls>
  <source src="test.mp4">
</video>




### Discussion

A few ideas to make the pipeline more robust: 
- Autodetection of the perspective transform: this will be necessary if the camera is ever moved from its original position. 
- Trying different color channels. I noticed that the lane detection tends to fail when the lighting changes (from light to shade, for example).


```python

```
