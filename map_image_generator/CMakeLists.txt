cmake_minimum_required(VERSION 3.8)

project(map_image_generator)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)

find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)
find_package(odas_ros_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    include/map_image_generator/drawers/GlobalPathImageDrawer.h
    include/map_image_generator/drawers/GoalImageDrawer.h
    include/map_image_generator/drawers/ImageDrawer.h
    include/map_image_generator/drawers/LabelImageDrawer.h
    include/map_image_generator/drawers/LaserScanImageDrawer.h
    include/map_image_generator/drawers/OccupancyGridImageDrawer.h
    include/map_image_generator/drawers/RobotImageDrawer.h
    include/map_image_generator/drawers/SoundSourceImageDrawer.h
    include/map_image_generator/GoalConverter.h
    include/map_image_generator/MapImageGenerator.h
    include/map_image_generator/MapLabelsConverter.h
    include/map_image_generator/Parameters.h
    include/map_image_generator/utils.h
    src/drawers/GlobalPathImageDrawer.cpp
    src/drawers/GoalImageDrawer.cpp
    src/drawers/ImageDrawer.cpp
    src/drawers/LabelImageDrawer.cpp
    src/drawers/LaserScanImageDrawer.cpp
    src/drawers/OccupancyGridImageDrawer.cpp
    src/drawers/RobotImageDrawer.cpp
    src/drawers/SoundSourceImageDrawer.cpp
    src/GoalConverter.cpp
    src/main.cpp
    src/MapImageGenerator.cpp
    src/MapLabelsConverter.cpp
    src/Parameters.cpp
    src/utils.cpp
)

ament_target_dependencies(${PROJECT_NAME} rclcpp std_msgs nav_msgs sensor_msgs geometry_msgs visualization_msgs rtabmap_msgs odas_ros_msgs opentera_webrtc_ros_msgs tf2 tf2_ros tf2_geometry_msgs cv_bridge image_transport)
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        "${OpenCV_INCLUDE_DIRS}"
)
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

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
