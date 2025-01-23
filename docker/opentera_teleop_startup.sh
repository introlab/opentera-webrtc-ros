export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/OPENTERA_WS/opentera-webrtc-ros/install/setup.bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib 
export LIBVA_DRIVER_NAME=nvidia
# Add the signaling server to PATH
export PATH=/opt/ros/OPENTERA_WS/opentera-webrtc-ros/install/opentera_webrtc_ros/local/bin:$PATH
ros2 pkg prefix opentera_client_ros
echo "PATH is: $PATH"
sh -c "chmod 0700 /tmp/runtime-root"

source /usr/share/gazebo/setup.sh
export PATH=$PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins  # For Gazebo plugins
export GAZEBO_MODEL_PATH=/opt/ros/OPENTERA_WS/opentera-webrtc-ros/install/opentera_webrtc_demos/share/opentera_webrtc_demos/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_PATH=/opt/ros/OPENTERA_WS/models:$GAZEBO_MODEL_PATH



#https://docs.ros.org/en/humble/How-To-Guides/DDS-tuning.html#cyclone-dds-tuning
sudo sysctl -w net.core.rmem_max=2147483647

echo $ROS_DISTRO
ros2 launch  startup_opentera_demos.launch.py
#ros2 run rviz2 rviz2