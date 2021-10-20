# opentera-webrtc-ros

Based on [opentera-webrtc](https://github.com/introlab/opentera-webrtc). We are now using a pre-built version 0.2.4 of our library to speed-up compilation.


# Authors

* Cédric Godin (@godced)
* Marc-Antoine Maheux (@mamaheux)
* Dominic Létourneau (@doumdi)
* Gabriel Lauzier (@G-Lauz)
* Jérémie Bourque (@JeremieBourque1)

# Requirements

The procedure is written for Ubuntu 20.04 using ROS noetic. We assume ROS is already installed. If not, follow the [ROS Installation Instructions](http://wiki.ros.org/noetic/Installation/Ubuntu) first. The following packages must also be installed :

```bash
# opentera-webrtc-ros packages
sudo apt-get install nodejs ros-noetic-turtlebot3 ros-noetic-turtlebot3-gazebo ros-noetic-dwa-local-planner ros-noetic-rtabmap-ros

# python dependencies
sudo apt-get install python3-pip portaudio19-dev

# nodejs dependencies
sudo apt-get install nodejs npm

# audio_utils packages
sudo apt-get install build-essential gfortran texinfo libasound2-dev
```


# Installation

## 1 - Install Qt 5.9.9 with Qt Maintenance Tool (required for newer versions of Ubuntu > 18.04 only)

Get the Online Installer [here](https://www.qt.io/download-thank-you?hsLang=en). You need to register with Qt to use the installer (or Maintenance Tool if you already installed Qt with the installer). Once registered, install Qt 5.9.9 for Linux by selecting :
* Desktop gcc 64 bits
* Sources
* Every Qt Modules (Android modules not needed)

>Make sure files are installed in the default location $HOME/Qt/5.9.9/gcc_64.

>We use Qt 5.9.9 since it is the version installed on Ubuntu 18.04 on our Jetson development boards.


## 2 - Create a catkin workspace (if not already done)

```bash
# Make sure ROS is installed first. 
$ source /opt/ros/noetic/setup.bash
# Create the workspace and initial build files
$ mkdir -p ~/teleop_ws/src
$ cd ~/teleop_ws/
$ catkin_make
```

## 3 - Get all the required ROS packages
```bash
$ cd ~/teleop_ws/src
# audio_utils
$ git clone https://github.com/introlab/audio_utils.git --recurse-submodules
# opentera-webrtc-ros
$ git clone https://github.com/introlab/opentera-webrtc-ros.git --recurse-submodules
```

## 4 - Build all the ROS packages
```bash
$ cd ~/teleop_ws
$ source devel/setup.bash
$ catkin_make
```

## 5 - Install the Python requirements for client and signaling server
```bash
$ cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_client_ros
$ python3 -m pip install -r requirements.txt
$ cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_webrtc_ros/opentera-webrtc/signaling-server
$ python3 -m pip install -r requirements.txt
```

## 6 - Install the VUE.js frontend (opentera-webrtc-teleop-frontend)
```bash
$ cd ~/teleop_ws/src/opentera-webrtc-ros/opentera_webrtc_demos/opentera-webrtc-teleop-frontend/teleop-vue
# Run the npm package installer
$ npm install
# Build the frontend (this will create a dist directory with all required files)
$ npm run build
```

# Running the demos

Please see the [opentera_webrtc_demos package.](opentera_webrtc_demos/README.md)


# License

* [Apache License, Version 2.0](LICENSE)

# Sponsor

![IntRoLab](images/IntRoLab.png)

[IntRoLab - Intelligent / Interactive / Integrated / Interdisciplinary Robot Lab](https://introlab.3it.usherbrooke.ca)

