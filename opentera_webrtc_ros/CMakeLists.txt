cmake_minimum_required(VERSION 3.1)
include(ExternalProject)
project(opentera_webrtc_ros)

## Set compile options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-fPIC")

if(NOT CMAKE_BUILD_TYPE )
    MESSAGE(STATUS "not define build type, set to release" )
    set(CMAKE_BUILD_TYPE Release )
elseif(NOT (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug" ))
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be either Release or Debug")
endif()

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    cv_bridge
    message_generation
    std_msgs
    sensor_msgs
    audio_utils
    geometry_msgs
    opentera_webrtc_ros_msgs
    std_srvs
    visualization_msgs
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

# This is required for WebRTC
add_definitions(-DWEBRTC_POSIX)

find_package(OpenCV REQUIRED core imgproc)

# Actually generate the language-specific message and service files
# generate_messages(
#   DEPENDENCIES
#   opentera_webrtc_ros_msgs
# )


###################################
## catkin specific configuration ##
###################################
catkin_package(CATKIN_DEPENDS message_runtime)


# Disable tests because of conflicts with gtest with catkin
set(OPENTERA_WEBRTC_ENABLE_TESTS OFF)
# Disable examples
set(OPENTERA_WEBRTC_ENABLE_EXAMPLES OFF)
# Use system opencv that is shared with ROS
set(OPENTERA_WEBRTC_USE_SYSTEM_OPENCV ON)
# Enable install on build for python package, which is required for using it in ROS if using devel space
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_INSTALL_ON_BUILD ON)

## !! Needs to be after catkin_package !!
# CATKIN_PACKAGE_PYTHON_DESTINATION and CATKIN_PACKAGE_BIN_DESTINATION are set by catkin_package
# Install to develspace on build
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_PIP_DEVEL_PREFIX ${CATKIN_DEVEL_PREFIX} CACHE STRING "")
# Install to installspace on install
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_PIP_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "")
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_PIP_PREFIX_USE_EXACT_LIB_SUBPATH "python3/dist-packages" CACHE STRING "")

execute_process(
    COMMAND python3 --version
    COMMAND bash -c "cut -f2 -d' '"
    OUTPUT_VARIABLE OPENTERA_WEBRTC_ROS_PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (${OPENTERA_WEBRTC_ROS_PYTHON_VERSION} VERSION_LESS "3.8.0")
    set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_PIP_PYTHON_EXECUTABLE_SIGNALING_SERVER "python3.8" CACHE STRING "")
endif ()

add_subdirectory(opentera-webrtc)

include_directories(
    ${webrtc_native_INCLUDE}
    ${boringssl_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/OpenteraWebrtcNativeClient/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/socket.io-client-cpp/src
)

set(opentera_webrtc_native_client_libs
    opencv_core
    opencv_imgproc
    OpenteraWebrtcNativeClient
    ${webrtc_native_LIBRARY}
    ${boringssl_LIBRARY}
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    lib
)


## Declare a C++ executable
add_executable(${PROJECT_NAME}_topic_streamer
    src/RosVideoSource.cpp
    src/RosAudioSource.cpp
    src/RosSignalingServerConfiguration.cpp
    src/RosStreamBridge.cpp
    src/RosNodeParameters.cpp
    src/RosParamUtils.cpp
)

add_executable(${PROJECT_NAME}_data_channel_bridge
    src/RosSignalingServerConfiguration.cpp
    src/RosDataChannelBridge.cpp
    src/RosNodeParameters.cpp
    src/RosParamUtils.cpp
)

add_executable(${PROJECT_NAME}_json_data_handler
    src/RosJsonDataHandler.cpp
)

## Rename C++ executable without prefix
set_target_properties(${PROJECT_NAME}_topic_streamer PROPERTIES OUTPUT_NAME topic_streamer PREFIX "")
set_target_properties(${PROJECT_NAME}_data_channel_bridge PROPERTIES OUTPUT_NAME data_channel_bridge PREFIX "")
set_target_properties(${PROJECT_NAME}_json_data_handler PROPERTIES OUTPUT_NAME json_data_handler PREFIX "")

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_topic_streamer
    ${opentera_webrtc_native_client_libs}
    ${catkin_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_data_channel_bridge
    ${opentera_webrtc_native_client_libs}
    ${catkin_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_json_data_handler
    ${opentera_webrtc_native_client_libs}
    ${catkin_LIBRARIES}
)

## ROS message dependencies
add_dependencies(${PROJECT_NAME}_topic_streamer ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME}_data_channel_bridge ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME}_json_data_handler ${catkin_EXPORTED_TARGETS})


#############
## Install ##
#############

install(
    PROGRAMS
        scripts/opentera_webrtc_audio_mixer.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# all install targets should use catkin DESTINATION variables

## Mark executables for installation
install(
    TARGETS
        ${PROJECT_NAME}_topic_streamer
        ${PROJECT_NAME}_data_channel_bridge
        ${PROJECT_NAME}_json_data_handler
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
