cmake_minimum_required(VERSION 3.0.2)
# include(ExternalProject)
project(opentera_webrtc_rosbridge_server)

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
  # roscpp
  rospy
  # cv_bridge
  message_generation
  std_msgs
  sensor_msgs
  # audio_utils
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


###########
## Build ##
###########

## Specify additional locations of header files
# include_directories(
#   include
#   ${catkin_INCLUDE_DIRS}
#   lib
# )


## Declare a C++ executable
# add_executable(${PROJECT_NAME}_topic_streamer
#   src/RosVideoSource.cpp
#   src/RosAudioSource.cpp
#   src/RosSignalingServerConfiguration.cpp
#   src/RosStreamBridge.cpp
#   src/RosNodeParameters.cpp
#   src/RosParamUtils.cpp
# )

# add_executable(${PROJECT_NAME}_data_channel_bridge
#   src/RosSignalingServerConfiguration.cpp
#   src/RosDataChannelBridge.cpp
#   src/RosNodeParameters.cpp
#   src/RosParamUtils.cpp
# )

# add_executable(${PROJECT_NAME}_json_data_handler
#   src/RosJsonDataHandler.cpp
# )

## Rename C++ executable without prefix
# set_target_properties(${PROJECT_NAME}_topic_streamer PROPERTIES OUTPUT_NAME topic_streamer PREFIX "")
# set_target_properties(${PROJECT_NAME}_data_channel_bridge PROPERTIES OUTPUT_NAME data_channel_bridge PREFIX "")
# set_target_properties(${PROJECT_NAME}_json_data_handler PROPERTIES OUTPUT_NAME json_data_handler PREFIX "")

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_topic_streamer
#   ${opentera_webrtc_native_client_libs}
#   ${catkin_LIBRARIES}
#   samplerate
# )

# target_link_libraries(${PROJECT_NAME}_data_channel_bridge
#   ${opentera_webrtc_native_client_libs}
#   ${catkin_LIBRARIES}
# )

# target_link_libraries(${PROJECT_NAME}_json_data_handler
#   ${opentera_webrtc_native_client_libs}
#   ${catkin_LIBRARIES}
# )

## ROS message dependencies
# add_dependencies(${PROJECT_NAME}_topic_streamer ${catkin_EXPORTED_TARGETS})
# add_dependencies(${PROJECT_NAME}_data_channel_bridge ${catkin_EXPORTED_TARGETS})
# add_dependencies(${PROJECT_NAME}_json_data_handler ${catkin_EXPORTED_TARGETS})

# set(PYTHON_THINGY_DIRECTORY ${PROJECT_BINARY_DIR}/python-thingy)

# add_custom_target(python-opentera-webrtc-native-client-so
#   ALL
#   DEPENDS _opentera_webrtc_native_client
#   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_opentera_webrtc_native_client> ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc/opentera-webrtc-native-client/OpenteraWebrtcNativeClient/python/package/opentera_webrtc_native_client/$<TARGET_FILE_NAME:_opentera_webrtc_native_client>
# )

# add_custom_target(python-opentera-webrtc-native-client-wheel
#   ALL
#   DEPENDS python-opentera-webrtc-native-client-so
#   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc/opentera-webrtc-native-client/OpenteraWebrtcNativeClient/python/package/opentera_webrtc_native_client ${PYTHON_THINGY_DIRECTORY}/
#   COMMAND /usr/bin/env python3.8 setup.py bdist bdist_wheel sdist
#   WORKING_DIRECTORY ${PYTHON_THINGY_DIRECTORY}
# )

# add_custom_target(python-opentera-webrtc-native-client
#   ALL
#   DEPENDS python-opentera-webrtc-native-client-wheel
#   COMMAND /usr/bin/env python3.8 -m pip install --upgrade --force-reinstall --no-deps --no-cache-dir --no-index --find-links=${PYTHON_THINGY_DIRECTORY}/dist opentera-webrtc-native-client
#   WORKING_DIRECTORY ${PYTHON_THINGY_DIRECTORY}
# )


#############
## Install ##
#############

# install(PROGRAMS
#    scripts/opentera_webrtc_audio_mixer.py
#       DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# all install targets should use catkin DESTINATION variables

## Mark executables for installation
# install(
#   TARGETS
#     ${PROJECT_NAME}_topic_streamer
#     ${PROJECT_NAME}_data_channel_bridge
#     ${PROJECT_NAME}_json_data_handler
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
