cmake_minimum_required(VERSION 3.8)
include(ExternalProject)
project(opentera_webrtc_ros)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE )
    MESSAGE(STATUS "not define build type, set to release" )
    set(CMAKE_BUILD_TYPE Release )
elseif(NOT (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug" ))
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be either Release or Debug")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(audio_utils REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)
find_package(map_image_generator REQUIRED)

find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED core imgproc)

add_definitions(-DWEBRTC_POSIX)

# Disable tests because of conflicts with gtest with catkin
set(OPENTERA_WEBRTC_ENABLE_TESTS OFF)
# Disable examples
set(OPENTERA_WEBRTC_ENABLE_EXAMPLES OFF)
# Enable GStreamer
set(OPENTERA_WEBRTC_ENABLE_GSTREAMER ON)
# Use system opencv that is shared with ROS
set(OPENTERA_WEBRTC_USE_SYSTEM_OPENCV ON)
# Enable install on build for python package, which is required for using it in ROS if using devel space
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_INSTALL_ON_BUILD OFF)
# Install in ROS worksapce
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_PIP_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "")

execute_process(
    COMMAND python3 --version
    COMMAND bash -c "cut -f2 -d' '"
    OUTPUT_VARIABLE OPENTERA_WEBRTC_ROS_PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (${OPENTERA_WEBRTC_ROS_PYTHON_VERSION} VERSION_LESS "3.8.0")
    set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_PIP_PYTHON_EXECUTABLE_SIGNALING_SERVER "python3.8" CACHE STRING "")
endif ()

# Use the system (ROS) version of Python
set(OPENTERA_WEBRTC_NATIVE_CLIENT_PYTHON_VERSION "${OPENTERA_WEBRTC_ROS_PYTHON_VERSION}" CACHE STRING "")

add_subdirectory(opentera-webrtc SYSTEM)

set(opentera_webrtc_native_client_includes
    ${webrtc_native_INCLUDE}
    ${boringssl_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/OpenteraWebrtcNativeClient/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/IXWebSocket
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/3rdParty/cpp-httplib
)
set(opentera_webrtc_native_client_targets
    OpenteraWebrtcNativeClient
)

if(OPENTERA_WEBRTC_ENABLE_GSTREAMER)
    list(APPEND opentera_webrtc_native_client_includes
        ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc/opentera-webrtc-native-client/OpenteraWebrtcNativeGStreamer/include
    )
    list(APPEND opentera_webrtc_native_client_targets
        OpenteraWebrtcNativeGStreamer
    )
endif()

set(opentera_webrtc_native_client_imported_libs
    opencv_core
    opencv_imgproc
    ${webrtc_native_LIBRARY}
    ${boringssl_LIBRARY}
)

set(opentera_webrtc_native_client_raw_libs ${opentera_webrtc_native_client_imported_libs})
list(FILTER opentera_webrtc_native_client_raw_libs INCLUDE REGEX "\\/.*")  # Only keep full paths, the rest are system libraries

set(opentera_webrtc_native_client_libs
    ${opentera_webrtc_native_client_targets}
    ${opentera_webrtc_native_client_imported_libs}
)

include_directories(SYSTEM
    ${opentera_webrtc_native_client_includes}
)

install(DIRECTORY ${opentera_webrtc_native_client_includes} DESTINATION include)
install(TARGETS ${opentera_webrtc_native_client_targets} DESTINATION lib)
install(FILES ${opentera_webrtc_native_client_raw_libs} DESTINATION lib)

include_directories(
    include
)
include_directories(SYSTEM
    lib
)

add_executable(${PROJECT_NAME}_topic_streamer
    src/RosVideoSource.cpp
    src/RosAudioSource.cpp
    src/RosSignalingServerConfiguration.cpp
    src/RosVideoStreamConfiguration.cpp
    src/RosStreamBridge.cpp
    src/RosNodeParameters.cpp
)
ament_target_dependencies(${PROJECT_NAME}_topic_streamer
    rclcpp
    map_image_generator
    opentera_webrtc_ros_msgs
    cv_bridge
    std_msgs
    std_srvs
    sensor_msgs
    geometry_msgs
    audio_utils
)
target_link_libraries(${PROJECT_NAME}_topic_streamer
    ${opentera_webrtc_native_client_libs}
)
set_target_properties(${PROJECT_NAME}_topic_streamer PROPERTIES OUTPUT_NAME topic_streamer PREFIX "")
install(TARGETS ${PROJECT_NAME}_topic_streamer DESTINATION lib/${PROJECT_NAME})

add_executable(${PROJECT_NAME}_data_channel_bridge
    src/RosSignalingServerConfiguration.cpp
    src/RosDataChannelBridge.cpp
    src/RosNodeParameters.cpp
)
ament_target_dependencies(${PROJECT_NAME}_data_channel_bridge
    rclcpp
    map_image_generator
    opentera_webrtc_ros_msgs
    cv_bridge
    std_msgs
    std_srvs
    sensor_msgs
    geometry_msgs
    audio_utils
)
target_link_libraries(${PROJECT_NAME}_data_channel_bridge
    ${opentera_webrtc_native_client_libs}
)
set_target_properties(${PROJECT_NAME}_data_channel_bridge PROPERTIES OUTPUT_NAME data_channel_bridge PREFIX "")
install(TARGETS ${PROJECT_NAME}_data_channel_bridge DESTINATION lib/${PROJECT_NAME})


add_executable(${PROJECT_NAME}_json_data_handler
    src/RosJsonDataHandler.cpp
)
ament_target_dependencies(${PROJECT_NAME}_json_data_handler
    rclcpp
    map_image_generator
    opentera_webrtc_ros_msgs
    cv_bridge
    std_msgs
    std_srvs
    sensor_msgs
    geometry_msgs
    audio_utils
)
set_target_properties(${PROJECT_NAME}_json_data_handler PROPERTIES OUTPUT_NAME json_data_handler PREFIX "")
install(TARGETS ${PROJECT_NAME}_json_data_handler DESTINATION lib/${PROJECT_NAME})

install(
    DIRECTORY
        lib/lib
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

install(
    PROGRAMS
        scripts/goal_manager.py
        scripts/labels_manager.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(
    ${opentera_webrtc_native_client_includes}
)
ament_export_libraries(
    ${opentera_webrtc_native_client_libs}
)

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
