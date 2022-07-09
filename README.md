# opentera-webrtc-ros

Based on [opentera-webrtc](https://github.com/introlab/opentera-webrtc).

# License
By default, libwebrtc is built with non-free codecs. See [opentera-webrtc](https://github.com/introlab/opentera-webrtc#license) to build without them.

# Authors

- Cédric Godin (@godced)
- Marc-Antoine Maheux (@mamaheux)
- Dominic Létourneau (@doumdi)
- Gabriel Lauzier (@G-Lauz)
- Jérémie Bourque (@JeremieBourque1)
- Philippe Warren (@philippewarren)

# Requirements

The procedure is written for Ubuntu 20.04 using ROS noetic. We assume ROS is already installed. If not, follow the [ROS Installation Instructions](http://wiki.ros.org/noetic/Installation/Ubuntu) first. The following packages must also be installed :

```bash
# opentera-webrtc-ros packages
sudo apt-get install nodejs ros-noetic-turtlebot3 ros-noetic-turtlebot3-gazebo ros-noetic-dwa-local-planner ros-noetic-rtabmap-ros

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
```

# Installation

## 1 - Create a catkin workspace (if not already done)

```bash
# Make sure ROS is installed first.
$ source /opt/ros/noetic/setup.bash
# Create the workspace and initial build files
$ mkdir -p ~/teleop_ws/src
$ cd ~/teleop_ws/
$ catkin_make
```

## 2 - Get all the required ROS packages

```bash
$ cd ~/teleop_ws/src
# audio_utils
$ git clone https://github.com/introlab/audio_utils.git --recurse-submodules
# odas_ros
$ git clone https://github.com/introlab/odas_ros.git --recurse-submodules
# opentera-webrtc-ros
$ git clone https://github.com/introlab/opentera-webrtc-ros.git --recurse-submodules
```

## 3 - Install the Python requirements

```bash
$ cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_client_ros
$ python3 -m pip install -r requirements.txt
$ cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_webrtc_ros
$ python3 -m pip install -r requirements.txt
$ cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_webrtc_ros/opentera-webrtc
$ python3 -m pip install -r requirements.txt
```

## 4 - Build all the ROS packages

```bash
$ cd ~/teleop_ws
$ source devel/setup.bash
$ catkin_make
```

# Running the demos

Please see the [opentera_webrtc_demos package.](opentera_webrtc_demos/README.md)

# License

- [Apache License, Version 2.0](LICENSE)

# Sponsor

![IntRoLab](images/IntRoLab.png)

[IntRoLab - Intelligent / Interactive / Integrated / Interdisciplinary Robot Lab](https://introlab.3it.usherbrooke.ca)
