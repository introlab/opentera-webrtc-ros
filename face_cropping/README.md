# face_cropping

## Installation

The following ROS packages are required:

- roscpp
- cv_bridge
- sensor_msgs
- opentera_webrtc_ros_msgs
- image_transport

## Description

Implements a ROS node that subscribes to an image stream. It detects a face in the received frame and crops it out. If more than one face is detected, it publishes the received image without changes. But if one face is detected it advertises the up or down scaled cropped image containing the face.

It can either subscribe to a `opentera_webrtc_ros::PeerImage` or a `sensor_msgs::Image` and advertises the same type depending on the `is_peer_image` parameter.

### Subscribes

- input_image : `opentera_webrtc_ros::PeerImage` or `sensor_msgs::Image`

### Advertises

- output_image : `opentera_webrtc_ros::PeerImage` or `sensor_msgs::Image`

For usage exemple look at [face_cropping.launch](launch/face_cropping.launch).

## Parameters

The face cropping node contains multiple parameters which can improve performance and experience depending on the background, the screen and the user.

### refresh_rate
The refresh rate of the user's screen, integer 30 fps by default.

### width and height
The width and height of the cropped face that are both integers, an important note is that the aspect ratio derived from these values is going to affect how smooth the movement feels. Since that aspect ratio needs to be preserved, the minimum steps of the frame surrounding the face are of that aspect ratio in pixels. Therefore an aspect ratio of 1:1 is going to feel smoother than one of 16:9, an aspect ratio of lower values is recommended. The default values are an height of 800 and a width of 400.

### seconds_without_detection
An integer representing the amount of seconds needed without a detection before the frame resets to show the whole image. A value too low, may cause unwanted zoom outs and one too high will cause the image to stay cropped even if no one is detected, by default as a value of 5.

### frames_used_for_stabilizer
The number of frames used to stabilize the frame surrounding the face, a value too small may cause the frame to be jittery, but one too high will cause a delay in the movement of the frame, by default 15.

### min_width_change, min_height_change, min_x_change, min_y_change
A double indicating the percentage of pixels needed to cause the movement of the frame surrounding the face. To put value that are too high will cause the face to be uncentered and maybe even out of the frame, but values that are too low will cause unwanted movement, by default all value of 15%.

### right_margin, left_margin, top_margin, bottom_margin
All doubles, represents the margins in percentage surrounding the face, an important note is that these values aren't absolute. The aspect ratio of the frame will precede these, so either the vertical or horizontal margins will be respected depending on the aspect ratio and the size of the detected face. By default right, left and top have values of 10% and the bottom 20%.

### is_peer_image
A boolean that shows if the image subscribes to and publishes a `opentera_webrtc_ros::PeerImage` or `sensor_msgs::Image`, by default tha value is false.

### haar_cascade_path and lbp_cascade_path
Both strings, the paths to the cascade files used to detect faces. By default this is the path to the files in the models folder where we installed said files, but other cascade files can be used.

### use_lbp
A boolean value that determines if haar or lbp cascades are used for face detection. Haar face detection is more accurate and causes less false detections, but lbp is faster and much more inaccurate, by default haar is used.

### detection_frames
An integer that represents when the detection occurs, for example a value of 3 will cause the program to detect the faces every 3 frames. This won't improve performance in terms of fps, but may cause a reduction in the cpu utilization. Going above 2 or 3 will cause the cropping to be less responsive. By default has a value of 1.

### detection_scale
A double that represents a percentage of the incoming image's resolution. It's used to downsacle the image before detecting the face causing major improvements in performance. The outgoing image will stay at full resolution, but the detection will use a lower resolution. Going below a resolution of 480p may cause problems with the accuracy and range of the face detection. If the camera's resolution is above 480p-720p, it is strongly recommended to downscale it to improve performance.

### min_face_width and min_face_height
Integers representing the minimum size of a face in pixels. This helps reducing false detection but reduces the range of the face detection. A lower value also significantly reduces performance, therefore a value in the range of 50 to 100 is recommended, by default 75x75.

### max_postion step and max_size_step
Doubles that indicate the minimum difference in percentage of the size or position between two detected faces before they're considered different ones, by default both have values of 15%.

### face_storing_frames
An integer representing the amount of frames where we keep a face in memory. Used with the parameter `valid_face_min_time`, it's used to reduce false detection by setting a minimum number of frames where a face has to be detected in order to be considered valid. This value will augment the detection delay of a new face or a lost one. By default, 15 frames.

### valid_face_min_time
A float that indicates the percentage of the last frames where a face needs to be detected in order to be considered valid. Used along with the parameter `face_storing_frames`, it reduces false detections, by default 80%. So keeping the last 15 frames, a face would have to be detected 12 frames to be a valid one. 

### highlight_detections
A boolean that represents if the bounding boxes of the detections are shown. Mostly used for debug and testing purposes, setting this to true can be very useful to check for false detections and detection losses, by default it is false and no boxes are shown. When the boxes are shown, a green box means that the detection is valid, a red one that it's invalid and therefore not taken into consideration and no box means that there were no detections.

## Known issues

### Faces in angles
The face detection has trouble detecting either faces that are in angle or turned slightly. That is related to the detection method used it happens with both haar and lbp. It isn't a problem if the detection stops for less than the parameter `seconds_without_detection`, but if the user moves during that time, the frame will lose them. 

### Multiple people
The cropping is supposed to stop if more than one user is detected. But because of the inaccuracy of haar and lbp, the detection will lose some of the people and might zoom in on one person and switch to another or even start flashing between the cropped and uncropped images. Adjusting the `face_storing_frames` and `valid_face_min_time` by lowering the time nedeed to be a valid face may help to keep the detections stable. 

### False detections
Because of haar and lbp's inaccuracy a lot of false detections will occur if a background is noisy of has a lot of reflections. That being said the current algorithm is quite good at filtering out detections that flash in for a couple frames. Especially if we adjust the `face_storing_frames` and `valid_face_min_time` by increasing the time needed to be a valid face. The real problem occurs when haar or lbp constantly and falsely detects faces. It then becomes impossible to differentiate the false detections from someone new walking into the image. Having a neutral background will also help reducing false detections. In a last result one could turn on `highlight_detections` and check for the position of the false detection to manually change the background.

### Performance 
Since face detection can be quite heavy on the cpu, the performance may be too low on certain devices. A number of parameters can be changed to improve this. One could increase the values of `min_face_width` and `min_face_height`, increase the percentage of `detection_scale`, reduce the amount of detections made by increasing `detection_frames` or use lbp by setting `use_lbp` to true. All these measure may optimize your performance but will reduce the accuracy and quality of the cropping.
