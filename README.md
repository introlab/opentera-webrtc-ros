# opentera-webrtc-ros

Welcome to the opentera-webrtc-ros project. The goal of the project is to provide useful ROS nodes to stream audio/video/data through Google's WebRTC library wrapped in [opentera-webrtc](https://github.com/introlab/opentera-webrtc). Wrappers are written in C++ and Python and are compatible with ROS1 (Noetic) at the moment. We use the [signaling server](https://github.com/introlab/opentera-webrtc/tree/main/signaling-server) implementation provided by opentera-webrtc.

Here are the key features:

* [ROS Messages](opentera_webrtc_ros_msgs) adding compatibility with ROS and [OpenTera protobuf protocol](https://github.com/introlab/opentera_messages) used in[opentera-teleop-service](https://github.com/introlab/opentera-teleop-service).

* [ROS Streaming nodes](opentera_webrtc_ros/README.md) capable of sending / receiving audio, video and data from WebRTC streams.

* Teleoperation with a [opentera-webrtc-teleop-frontend](https://github.com/introlab/opentera-webrtc-teleop-frontend) sending and receiving robot commands from the WebRTC data chanel in JSON format. We hope to be compatible with [rosbridge_suite](https://github.com/RobotWebTools/rosbridge_suite) in the future.

* Map-based 2D/3D teleoperation using [RTAB-Map ROS](https://github.com/introlab/rtabmap_ros). Generated floor maps are streamed like a 2D image for simplicity at the moment by the [map_image_generator](map_image_generator) node.

* Sound Source Localization / Tracking / Separation using [ODAS ROS](https://github.com/introlab/odas_ros).

* [Stand alone demonstrations](opentera_webrtc_demos/README.md) with simulated robot in gazebo.

* [Robot side front-end (Qt)](opentera_webrtc_robot_gui/README.md) to display call information and remote user interaction.

* [OpenTera client](opentera_client_ros/README.md) basic implementation to act as a connected "device" that can receive calls from the OpenTera microservice cloud. A prototype web service (portal) is available in the [opentera-teleop-service](https://github.com/introlab/opentera-teleop-service) project.

> Note: This project is under developement. Any contribution or suggestion is welcome. Please use GitHub's [Issues](https://github.com/introlab/opentera-webrtc-ros/issues) system for questions or bug reports.

## License

By default, libwebrtc is built with non-free codecs. See [opentera-webrtc](https://github.com/introlab/opentera-webrtc#license) to build without them.

The project is licensed with:

* [Apache License, Version 2.0](LICENSE)

## Authors

* Cédric Godin (@godced)
* Marc-Antoine Maheux (@mamaheux)
* Dominic Létourneau (@doumdi)
* Gabriel Lauzier (@G-Lauz)
* Jérémie Bourque (@JeremieBourque1)
* Philippe Warren (@philippewarren)
* Ian-Mathieu Joly (@joli-1801)

## Dependencies / Requirements

The procedure is written for Ubuntu 20.04 using ROS noetic. We assume ROS is already installed. If not, follow the [ROS Installation Instructions](http://wiki.ros.org/noetic/Installation/Ubuntu) first. The following packages must also be installed :

```bash
# opentera-webrtc-ros packages
sudo apt-get install nodejs ros-noetic-turtlebot3 ros-noetic-turtlebot3-gazebo ros-noetic-cv-camera ros-noetic-dwa-local-planner ros-noetic-rtabmap-ros

# protobuf
sudo apt-get install libprotobuf-dev protobuf-compiler python3-protobuf

# python dependencies
sudo apt-get install python3-pip portaudio19-dev

# nodejs dependencies
sudo apt-get install nodejs npm

# audio_utils packages
sudo apt-get install cmake build-essential gfortran texinfo libasound2-dev libpulse-dev libgfortran-*-dev

# odas_ros packages
sudo apt-get install libfftw3-dev libconfig-dev

# qt submodules
sudo apt-get install libqt5charts5-dev
```

## Installation

### 1 - Create a catkin workspace (if not already done)

```bash
# Make sure ROS is installed first.
source /opt/ros/noetic/setup.bash
# Create the workspace and initial build files
mkdir -p ~/teleop_ws/src
cd ~/teleop_ws/
catkin_make
```

### 2 - Get all the required ROS packages

```bash
cd ~/teleop_ws/src
# audio_utils
git clone https://github.com/introlab/audio_utils.git --recurse-submodules
# odas_ros
git clone https://github.com/introlab/odas_ros.git --recurse-submodules
# opentera-webrtc-ros
git clone https://github.com/introlab/opentera-webrtc-ros.git --recurse-submodules
```

### 3 - Install the Python requirements

```bash
cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_client_ros
python3 -m pip install -r requirements.txt
cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_webrtc_ros
python3 -m pip install -r requirements.txt
cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_webrtc_ros/opentera-webrtc
python3 -m pip install -r requirements.txt
```

### 4 - Build all the ROS packages

```bash
cd ~/teleop_ws
source devel/setup.bash
catkin_make
```

## Running the demos

Please see the [opentera_webrtc_demos package.](opentera_webrtc_demos/README.md)

## Sponsor

![IntRoLab](https://introlab.3it.usherbrooke.ca/IntRoLab.png)

[IntRoLab - Intelligent / Interactive / Integrated / Interdisciplinary Robot Lab](https://introlab.3it.usherbrooke.ca)
