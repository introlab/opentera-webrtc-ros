cmake_minimum_required(VERSION 3.0.2)
include(ExternalProject)
project(opentera_webrtc_ros)

## Set compile options

add_compile_options(-std=c++14)
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

## Fetch latest opentera-webrtc-teleop build
set(opentera_webrtc_native_version "0.1.1")
set(opentera_webrtc_native_archive "opentera-webrtc-native-client-Linux-Ubuntu-20.04-x86_64-0.1.1.tar.gz")
set(opentera_webrtc_native_client_url "https://github.com/introlab/opentera-webrtc-teleop/releases/download/0.1.1/opentera-webrtc-native-client-Linux-Ubuntu-20.04-x86_64-0.1.1.tar.gz")
set(opentera_webrtc_native_archive_hash "3016ba2a1c4550661b8f9256bd894217")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${opentera_webrtc_native_archive})
    file(DOWNLOAD ${opentera_webrtc_native_client_url} ${CMAKE_CURRENT_BINARY_DIR}/${opentera_webrtc_native_archive}
        TIMEOUT 60 #seconds
        EXPECTED_MD5 ${opentera_webrtc_native_archive_hash}
        TLS_VERIFY ON)
endif()

# Extract archive
add_custom_target(
    extracted_webrtc_native_client
    COMMAND tar xvzf ${opentera_webrtc_native_archive}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc-native-client-${opentera_webrtc_native_version}/include
    ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc-native-client-${opentera_webrtc_native_version}/include/opencv4
    ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc-native-client-${opentera_webrtc_native_version}/include/rapidjson
)

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc-native-client-${opentera_webrtc_native_version}/lib
    ${CMAKE_CURRENT_BINARY_DIR}/opentera-webrtc-native-client-${opentera_webrtc_native_version}/lib/opencv4/3rdparty/lib              
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
  sioclient_tls
  webrtc boringssl
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
add_dependencies(${PROJECT_NAME}_topic_streamer extracted_webrtc_native_client)

add_executable(${PROJECT_NAME}_data_channel_bridge
  src/RosSignalingServerConfiguration.cpp
  src/RosDataChannelBridge.cpp
)
add_dependencies(${PROJECT_NAME}_data_channel_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} extracted_webrtc_native_client)

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

