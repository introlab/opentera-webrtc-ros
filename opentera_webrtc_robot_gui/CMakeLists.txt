cmake_minimum_required(VERSION 3.8)
project(opentera_webrtc_robot_gui)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Ubuntu release version
# find_program(LSB_RELEASE_EXEC lsb_release)
# execute_process(COMMAND ${LSB_RELEASE_EXEC} -rs
#     OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)

find_package(ament_index_cpp REQUIRED)

set(opentera_webrtc_robot_gui_components Core Widgets Gui OpenGL Charts)

# Uncomment if you want to use another Qt Version
# set(Qt5_DIR $ENV{HOME}/Qt/5.12.12/gcc_64/lib/cmake/Qt5)
find_package(Qt5 COMPONENTS ${opentera_webrtc_robot_gui_components} REQUIRED)

set(srcs
    src/main.cpp
    src/MainWindow.cpp
    src/ROSCameraView.cpp
    src/ConfigDialog.cpp
    src/Statistics.cpp
    src/LocalCameraWindow.cpp
)

set(headers
    src/MainWindow.h
    src/ROSCameraView.h
    src/ConfigDialog.h
    src/Statistics.h
    src/LocalCameraWindow.h
)

set(uis
    src/ui/MainWindow.ui
    src/ui/ConfigDialog.ui
    src/ui/Statistics.ui
)

set(qrcs
    src/resources/opentera_webrtc_robot_gui.qrc
)

#Generate .h files from the .ui files
QT5_WRAP_UI(project_moc_uis ${uis})

# generate rules for building source files from the resources
QT5_ADD_RESOURCES(project_qrcs ${qrcs})

#This will generate moc_* for Qt
qt5_wrap_cpp(project_moc_srcs ${headers})

add_executable(${PROJECT_NAME}_node ${srcs} ${uis} ${headers} ${project_moc_uis} ${project_qrcs} ${project_moc_srcs})
qt5_use_modules(${PROJECT_NAME}_node ${opentera_webrtc_robot_gui_components})
ament_target_dependencies(${PROJECT_NAME}_node rclcpp std_msgs sensor_msgs opentera_webrtc_ros_msgs ament_index_cpp)
target_link_libraries(${PROJECT_NAME}_node
    ${opentera_webrtc_robot_gui_components_link}
)
target_include_directories(
    ${PROJECT_NAME}_node
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})

install(FILES
    src/resources/DeviceProperties.json
    DESTINATION share/${PROJECT_NAME}/resources
)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)

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
