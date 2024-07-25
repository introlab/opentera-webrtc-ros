# opentera-webrtc-ros

Welcome to the opentera-webrtc-ros project. The goal of the project is to provide useful ROS nodes to stream audio/video/data through Google's WebRTC library wrapped in [opentera-webrtc](https://github.com/introlab/opentera-webrtc). Wrappers are written in C++ and Python and are compatible with ROS2 (Humble) at the moment. Support for ROS1 (Noetic) is in the [`ros1`](https://github.com/introlab/opentera-webrtc-ros/tree/ros1) branch. We use the [signaling server](https://github.com/introlab/opentera-webrtc/tree/main/signaling-server) implementation provided by opentera-webrtc.

Here are the key features:

* [ROS Messages](opentera_webrtc_ros_msgs) adding compatibility with ROS and [OpenTera protobuf protocol](https://github.com/introlab/opentera_messages) used in [opentera-teleop-service](https://github.com/introlab/opentera-teleop-service).

* [ROS Streaming nodes](opentera_webrtc_ros/README.md) capable of sending / receiving audio, video and data from WebRTC streams.

* Teleoperation with a [opentera-webrtc-teleop-frontend](https://github.com/introlab/opentera-webrtc-teleop-frontend) sending and receiving robot commands from the WebRTC data chanel in JSON format. We hope to be compatible with [rosbridge_suite](https://github.com/RobotWebTools/rosbridge_suite) in the future.

* Map-based 2D/3D teleoperation using [RTAB-Map ROS](https://github.com/introlab/rtabmap_ros). Generated floor maps are streamed like a 2D image for simplicity at the moment by the [map_image_generator](map_image_generator) node.

* Sound Source Localization / Tracking / Separation using [ODAS ROS](https://github.com/introlab/odas_ros).

* [Stand alone demonstrations](opentera_webrtc_demos/README.md) with simulated robot in Gazebo.

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

The procedure is written for Ubuntu 22.04 using ROS2 Humble. We assume ROS is already installed. If not, follow the [ROS2 Humble Installation Instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) first. A more recent CMake than the default on Ubunuttu 22.04 is required. If the installed CMake version is 3.22, follow the [CMake Installation Instructions](https://apt.kitware.com/). The following packages must also be installed :

```bash
# utilities
sudo apt install unzip rsync ros-dev-tools

# opentera-webrtc-ros packages
sudo apt install ros-humble-camera-info-manager ros-humble-rtabmap-ros ros-humble-rqt-tf-tree ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description ros-humble-turtlebot3-navigation2 ros-humble-joint-state-publisher-gui

# protobuf
sudo apt install libprotobuf-dev protobuf-compiler python3-protobuf

# python dependencies
sudo apt install python3-pip portaudio19-dev

# nodejs dependencies
sudo apt install nodejs npm

# audio_utils packages
sudo apt install cmake build-essential gfortran texinfo libasound2-dev libpulse-dev 'libgfortran-*-dev'

# odas_ros packages
sudo apt install libfftw3-dev libconfig-dev

# gstreamer for hardware acceleration
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools

# qt submodules
sudo apt install libqt5charts5-dev
```

## Installation

### 1 - Create a colcon workspace (if not already done)

```bash
# Create the workspace
mkdir -p ~/teleop_ws/src
```

### 2 - Get all the required ROS packages

```bash
cd ~/teleop_ws/src
# cv_camera
git clone https://github.com/Kapernikov/cv_camera.git
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
colcon build --symlink-install --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Debug --no-warn-unused-cli
```

## Running the demos

Please see the [opentera_webrtc_demos package.](opentera_webrtc_demos/README.md)

## Sponsor

![IntRoLab](https://introlab.3it.usherbrooke.ca/IntRoLab.png)

[IntRoLab - Intelligent / Interactive / Integrated / Interdisciplinary Robot Lab](https://introlab.3it.usherbrooke.ca)
