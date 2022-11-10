# face_cropping

## Installation

The following ROS packages are required:

- roscpp
- cv_bridge
- sensor_msgs
- opentera_webrtc_ros_msgs
- image_transport

### Description

Implements a ROS node that subscribes to an image stream. It detects a face in the received frame and crops it out. If more than one face is detected, it publishes the received image without changes. But if one face is detected it advertises the up or down scaled cropped image containing the face.

It can either subscribe to a `opentera_webrtc_ros::PeerImage` or a `sensor_msgs::ImageConstPtr` and advertises the same type depending on the `is_peer_image` parameter.

#### Subscribes

- input_image : `opentera_webrtc_ros::PeerImage` or `sensor_msgs::ImageConstPtr`

#### Advertises

- output_image : `opentera_webrtc_ros::PeerImage` or `sensor_msgs::ImageConstPtr`

For usage exemple look at [face_cropping.launch](launch/face_cropping.launch).
