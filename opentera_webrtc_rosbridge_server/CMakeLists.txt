cmake_minimum_required(VERSION 3.0.2)
project(opentera_webrtc_rosbridge_server)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  rosbridge_library
  rosbridge_msgs
  rosapi
  rosauth
  opentera_webrtc_ros_msgs
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

# This is required for WebRTC
# add_definitions(-DWEBRTC_POSIX)

# Disable tests because of conflicts with gtest with catkin
# set(OPENTERA_WEBRTC_ENABLE_TESTS OFF)
# Use system opencv that is shared with ROS
# set(OPENTERA_WEBRTC_USE_SYSTEM_OPENCV ON)
# Disable generation of Python doc, this is causing problems with catkin otherwise
# set(OPENTERA_WEBRTC_ENABLE_PYTHON_HTML_DOC OFF)
# add_subdirectory(opentera-webrtc/opentera-webrtc-native-client)

# find_package(OpenCV REQUIRED core imgproc)


# TODO Fixme. Version is hard coded for now
# set(OPENTERA_WEBRTC_NATIVE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/webrtc_native/webrtc-native-build-${webrtc_native_build_version}/include)
# set(OPENTERA_WEBRTC_NATIVE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/webrtc_native/webrtc-native-build-${webrtc_native_build_version}/lib)

# include_directories(
#     ${OPENTERA_WEBRTC_NATIVE_INCLUDE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/OpenteraWebrtcNativeClient/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty
#     ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/socket.io-client-cpp/src
# )

# link_directories(
#     ${OPENTERA_WEBRTC_NATIVE_LIB_DIR}
#     ${OPENTERA_WEBRTC_NATIVE_LIB_DIR}/boringssl_asm
# )

# set (opentera_webrtc_native_client_libs
#   opencv_core
#   opencv_imgproc
#   OpenteraWebrtcNativeClient
#   boringssl
#   webrtc
#   dl
#   X11
#   expat
# )

# Actually generate the language-specific message and service files
# generate_messages(
#   DEPENDENCIES
#   opentera_webrtc_ros_msgs
# )


###################################
## catkin specific configuration ##
###################################
catkin_package(CATKIN_DEPENDS message_runtime)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# Mark executable scripts (Python etc.) for installation
# in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  scripts/rosbridge_ros_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  launch/rosbridge_webrtc.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
