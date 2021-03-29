# Installation

The following ROS packages are required:
* roscpp
* cv_bridge
* std_msgs
* sensor_msgs

Also add the following [repository](https://github.com/introlab/audio_utils) in the catkin workspace src directory:
```bash
git clone https://github.com/introlab/audio_utils.git --recurse-submodules
```
See https://github.com/introlab/audio_utils for more informations about the dependency and the usage;


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

Implement a ROS node that publish received images as a WebRTC stream.
It also forwards images received on the WebRTC stream to ROS.

For now the node only handle a single video track in each direction and no audio.

### Subscribes

* ros_image : `sensor_msgs::Image`

### Advertises

* webrtc_image : `sensor_msgs::Image`

## Default Parameters

```yaml
~stream:
  is_screen_cast: false     # Is the image source a screen capture?
  needs_denoising: false    # Does the image source needs denoising?

~signaling:
  server_url: "http://localhost:8080" # Signaling server URL
  client_name: "streamer" # Peer name as which to join the room
  room_name: "chat"       # Room name to join
  room_password: "abc"    # Room password
```

# RosDataChannelBridge

## Description

Implement a ROS node that publish received messages on the WebRTC
data channel. It also forwards messages received on the WebRTC data channel to ROS.

### Subscribes

* ros_data : `std_msgs::String`

### Advertises

* webrtc_data : `std_msgs::String`

## Default Parameters

```yaml
~signaling:
  server_url: "http://localhost:8080" # Signaling server URL
  client_name: "data_bridge" # Peer name as which to join the room
  room_name: "chat"       # Room name to join
  room_password: "abc"    # Room password
```
