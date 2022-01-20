# Installation

The following ROS packages are required:

- roscpp
- cv_bridge
- std_msgs
- sensor_msgs

Also add the following [repository](https://github.com/introlab/audio_utils) in the catkin workspace src directory:

```bash
git clone https://github.com/introlab/audio_utils.git --recurse-submodules
```

See https://github.com/introlab/audio_utils for more informations about the dependency and his usage;

# Clone the repository with the submodules in the catkin workspace src directory

```bash
git clone https://github.com/introlab/opentera-webrtc-ros.git --recurse-submodules
```

# TODO

Complete the following documentation.

Finally the OpenTera WebRTC native client and its dependencies must have been built with same build type, Debug or
Release as the desired build output.

# RosStreamBridge

## Description

Implement a ROS node that publish received images and audio as a WebRTC stream.
It also forwards images and audio received on the WebRTC stream to ROS.

### Subscribes

- ros_image : `sensor_msgs::Image`

### Advertises

- webrtc_image : `opentera_webrtc_ros::PeerImage`
- webrtc_audio : `opentera_webrtc_ros::PeerAudio`
- audio_mixed : `audio_utils::AudioFrame`

## Default Parameters

```xml
<rosparam param="is_stand_alone">true</rosparam>
<rosparam param="stream">
  {
    can_send_audio_stream: true,        # Does the node can send audio tream to the signaling server
    can_receive_audio_stream: true,     # Does the node can receive audio stream from the signaling server
    can_send_video_stream: true,        # Does the node can send video stream to the signaling server
    can_receive_video_stream: true,     # Does the node can receive video stream from the signaling server
    is_screen_cast: false,        # Is the image source a screen capture?
    needs_denoising: false        # Does the image source needs denoising?
  }
</rosparam>
<rosparam param="signaling">
  {
    server_url: "http://localhost:8080",    # Signaling server URL
    client_name: "streamer",                # Peer name as which to join the room
    room_name: "chat",                      # Room name to join
    room_password: "abc"                    # Room password
  }
</rosparam>
```

For usage exemple look at [ros_stream_client.launch](launch/ros_stream_client.launch).

# RosDataChannelBridge

## Description

Implement a ROS node that publish received messages on the WebRTC
data channel. It also forwards messages received on the WebRTC data channel to ROS.

### Subscribes

- ros_data : `std_msgs::String`

### Advertises

- webrtc_data : `opentera_webrtc_ros_msgs::PeerData`

## Default Parameters

```xml
<rosparam param="is_stand_alone" >true</rosparam>
<rosparam param="signaling">
  {
    server_url: "http://localhost:8080",    # Signaling server URL
    client_name: "data_bridge",             # Peer name as which to join the room
    room_name: "chat",                      # Room name to join
    room_password: "abc"                    # Room password
  }
</rosparam>
```

For usage exemple look at [ros_data_channel_client.launch](launch/ros_data_channel_client.launch).

# RosJsonDataHandler

## Decription

Implement a ROS node that dispatch received JSON messages and forward them on the rights topics.

### Subscribes

- webrtc_data : `opentera_webrtc_ros_msgs::PeerData`

### Advertises

- cmd_vel : `geometry_msgs::Twist`

## JSON Format (TODO)

For usage exemple look at [ros_json_data_handler.launch](launch/ros_json_data_handler.launch).

# goal_manager

## Description

Manages multiple waypoints received by the frontend and sends them to move_base one at a time. This node relies a service provided by the `map_image_generator` package to convert the waypoints from image coordinates to map coordinates.

## Subscribed topics

- waypoints (`opentera_webrtc_ros_msgs/Waypoints`): Array of image coordinate waypoints received from the frontend.
- stop (`std_msgs/Bool`): Signal to cancel all move_base goals.

## Published topics

- waypoint_reached (`std_msgs/String`): String of a JSON message containing the ID of the waypoint that has been reached. Used by the frontend to determine when the trajectory has been completed.

# labels_manager

## Description

Manages labels.
The stored labels are dependent on the `map` transform and the database needs to be cleaned if the map changes.
A label represents a name grouped with an associated pose and a description.
This node relies on a service provided by the `map_image_generator` package to convert the labels from image coordinates to map coordinates.

<!-- This node can also send a label as a goal to `move_base` by its name. -->

## Subscribed topics

- add_label_simple (`opentera_webrtc_ros_msgs/LabelSimple`): Label received from the frontend.
- remove_label_by_name (`std_msgs/String`): Remove a label by its name
- edit_label_simple (`opentera_webrtc_ros_msgs/LabelSimpleEdit`): Rename or move a label using frontend map coordinates

## Published topics

- stored_labels (`opentera_webrtc_ros_msgs/LabelArray`): Array of labels currently stored
<!-- - stored_labels_simple (`openterat_webrtc_ros_msgs/LabelSimpleArray`): Version of `stored_labels` which contains `opentera_webrtc_ros_msgs/Waypoint` instead of `geometry_msgs/PoseStamped` for frontend communication. -->
