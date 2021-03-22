cmake_minimum_required(VERSION 3.0.2)
include(ExternalProject)
project(opentera_webrtc_ros)

## Set compile options

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-fPIC")
add_compile_definitions(WEBRTC_POSIX)

if(NOT CMAKE_BUILD_TYPE )
  MESSAGE(STATUS "not define build type, set to release" )
  set(CMAKE_BUILD_TYPE Release )
elseif(NOT (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug" ))
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be either Release or Debug")
endif()

## Find catkin macros and libraries

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  message_generation
  std_msgs
  sensor_msgs
)

# You must set the environment variable
# OPENTERA_WEBRTC_DIST_DIR where opentera-webrtc is installed
if(NOT DEFINED ENV{OPENTERA_WEBRTC_DIST_DIR})
    message(FATAL_ERROR "Environment variable OPENTERA_WEBRTC_DIST_DIR must be set.")
endif()

# TODO version is hard coded for now
set(OPENTERA_WEBRTC_INCLUDE_DIR $ENV{OPENTERA_WEBRTC_DIST_DIR}/${CMAKE_BUILD_TYPE}/opentera-webrtc-native-client-0.2.2/include)
set(OPENTERA_WEBRTC_LIB_DIR $ENV{OPENTERA_WEBRTC_DIST_DIR}/${CMAKE_BUILD_TYPE}/opentera-webrtc-native-client-0.2.2/lib)

# TODO UGLY!
set(OPENTERA_WEBRTC_NATIVE_INCLUDE_DIR $ENV{OPENTERA_WEBRTC_DIST_DIR}/../3rdParty/webrtc_native/webrtc-native-build-4389.742109.0/include)
set(OPENTERA_WEBRTC_NATIVE_LIB_DIR $ENV{OPENTERA_WEBRTC_DIST_DIR}/../3rdParty/webrtc_native/webrtc-native-build-4389.742109.0/lib)


message(STATUS "**************************** ${OPENTERA_WEBRTC_INCLUDE_DIR} ${OPENTERA_WEBRTC_LIB_DIR}")

include_directories(
    ${OPENTERA_WEBRTC_INCLUDE_DIR}
    ${OPENTERA_WEBRTC_NATIVE_INCLUDE_DIR}
)

link_directories(
    ${OPENTERA_WEBRTC_LIB_DIR}
    ${OPENTERA_WEBRTC_LIB_DIR}/opencv4/3rdparty
    ${OPENTERA_WEBRTC_NATIVE_LIB_DIR}      
)

set (opentera_webrtc_native_client_libs
  opencv_core 
  opencv_imgproc
  ade 
  ippicv 
  ippiw 
  ittnotify   
  libopenjp2  
  zlib
  OpenteraWebrtcNativeClient
  sioclient
  webrtc 
  boringssl
  dl 
  X11 
  expat
)


## System dependencies are found with CMake's conventions
## find_package(OpenCV REQUIRED core imgproc)

# if(NOT DEFINED ENV{OPEN_TERA_WEBRTC_NATIVE_CLIENT_ROOT_DIR})
#   message(FATAL_ERROR "OPEN_TERA_WEBRTC_NATIVE_CLIENT_ROOT_DIR environnement variable must be configured.")
# endif()

# if(NOT DEFINED ENV{OPEN_TERA_WEBRTC_NATIVE_CLIENT_LIB})
#   message(FATAL_ERROR "OPEN_TERA_WEBRTC_NATIVE_CLIENT_LIB environnement variable must be configured.")
# endif()


##############################
## Generate custom messages ##
##############################

# Declare the message files to be built
add_message_files(FILES
  Peer.msg
  PeerData.msg
)

# Actually generate the language-specific message and service files
generate_messages(DEPENDENCIES std_msgs)


###################################
## catkin specific configuration ##
###################################

catkin_package(CATKIN_DEPENDS message_runtime)


###########
## Build ##
###########

## Specify additional locations of header files

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Specify additional locations of library files



## Declare a C++ executable

add_executable(${PROJECT_NAME}_topic_streamer
  src/RosVideoSource.cpp
  src/RosAudioSource.cpp
  src/RosSignalingServerConfiguration.cpp
  src/RosStreamBridge.cpp
)
# add_dependencies(${PROJECT_NAME}_topic_streamer extracted_webrtc_native_client)

add_executable(${PROJECT_NAME}_data_channel_bridge
  src/RosSignalingServerConfiguration.cpp
  src/RosDataChannelBridge.cpp
)
# add_dependencies(${PROJECT_NAME}_data_channel_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} extracted_webrtc_native_client)

## Rename C++ executable without prefix

set_target_properties(${PROJECT_NAME}_topic_streamer PROPERTIES OUTPUT_NAME topic_streamer PREFIX "")
set_target_properties(${PROJECT_NAME}_data_channel_bridge PROPERTIES OUTPUT_NAME data_channel_bridge PREFIX "")

## Specify libraries to link a library or executable target against

target_link_libraries(${PROJECT_NAME}_topic_streamer
  ${catkin_LIBRARIES}
  ${opentera_webrtc_native_client_libs}
)

target_link_libraries(${PROJECT_NAME}_data_channel_bridge
  ${catkin_LIBRARIES}
  ${opentera_webrtc_native_client_libs}
)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables

## Mark executables for installation

install(
  TARGETS
    ${PROJECT_NAME}_topic_streamer
    ${PROJECT_NAME}_data_channel_bridge
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

