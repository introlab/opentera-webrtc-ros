#set rmw and source workspace
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/OPENTERA_WS/opentera-webrtc-ros/install/setup.bash

# Add /usr/lib/x86_64-linux-gnu to the library path
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Add /usr/local/lib to the library path
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib 

# Set the Video Acceleration API (VA-API) driver to NVIDIA
# export LIBVA_DRIVER_NAME=nvidia

# Add the signaling server to PATH
export PATH=/opt/ros/OPENTERA_WS/opentera-webrtc-ros/install/opentera_webrtc_ros/local/bin:$PATH

# source gazebo and add additional gazebo model paths
source /usr/share/gazebo/setup.sh
export PATH=$PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins  # For Gazebo plugins
export GAZEBO_MODEL_PATH=/opt/ros/OPENTERA_WS/opentera-webrtc-ros/install/opentera_webrtc_demos/share/opentera_webrtc_demos/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_PATH=/opt/ros/OPENTERA_WS/models:$GAZEBO_MODEL_PATH



#https://docs.ros.org/en/humble/How-To-Guides/DDS-tuning.html#cyclone-dds-tuning
sudo sysctl -w net.core.rmem_max=2147483647

echo $ROS_DISTRO
ros2 launch startup_opentera_demos.launch.py
