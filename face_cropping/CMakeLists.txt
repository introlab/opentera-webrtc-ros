cmake_minimum_required(VERSION 3.16.3)
project(face_cropping)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  image_transport
  sensor_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

option(FACE_CROPPER_USE_CUDA "Add the support of CUDA" OFF)

# Jetson platforms use libtorch installed from the NVIDIA forum
if (EXISTS "/etc/nv_tegra_release")
    execute_process(COMMAND python3 -c "import torch;" RESULT_VARIABLE IMPORT_TORCH_EXIT_CODE OUTPUT_QUIET)

    if (${IMPORT_TORCH_EXIT_CODE} EQUAL 0)
        message(STATUS "PyTorch is found.")

        execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
                OUTPUT_VARIABLE CMAKE_PREFIX_PATH)

        find_package(Torch REQUIRED)
        find_package(TorchVision QUIET)
        if (TorchVision_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
            set(TORCHVISION_LIBRARIES TorchVision::TorchVision)
            include_directories(${TorchVision_INCLUDE_DIR})
        else()
            message(WARNING "TorchVision is not found.")
            add_compile_definitions(NO_TORCH)
        endif()
    else()
        message(WARNING "PyTorch and torchvision are not found.")
        add_compile_definitions(NO_TORCH)
    endif()
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
    if (FACE_CROPPER_USE_CUDA)
        find_package(CUDA) #See https://gist.github.com/ksopyla/bf74e8ce2683460d8de6e0dc389fc7f5
    endif()

    if (CUDA_FOUND)
        message(STATUS "CUDA ${CUDA_VERSION} found.")
        set(WITH_CUDA ON)
        if (${CUDA_VERSION} EQUAL 11.6)
            set(TORCH_URL "https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip")
        elseif(${CUDA_VERSION} EQUAL 11.7)
            set(TORCH_URL "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip")
        else()
            message(FATAL "CUDA ${CUDA_VERSION} is not supported.")
        endif()
    else()
        set(WITH_CUDA OFF)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip")
    endif()

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/libtorch)
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/libtorch/libtorch.zip)
        file(DOWNLOAD ${TORCH_URL} ${CMAKE_BINARY_DIR}/libtorch/libtorch.zip SHOW_PROGRESS)
    endif()
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/libtorch/libtorch/build-hash)
        execute_process(
                COMMAND unzip libtorch.zip
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/libtorch/
                RESULT_VARIABLE UNZIP_RETURN_CODE
        )
        if(NOT ${UNZIP_RETURN_CODE} EQUAL 0)
            message(FATAL_ERROR "unzip failed")
        endif()
    endif()
    set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch/libtorch)

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/torchvision)
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/torchvision/torchvision.zip)
        file(DOWNLOAD https://github.com/pytorch/vision/archive/refs/tags/v0.14.1.zip ${CMAKE_BINARY_DIR}/torchvision/torchvision.zip SHOW_PROGRESS)
    endif()
    if (NOT EXISTS ${CMAKE_BINARY_DIR}/torchvision/vision-0.14.1/version.txt)
        execute_process(
                COMMAND unzip torchvision.zip
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/torchvision/
                RESULT_VARIABLE UNZIP_RETURN_CODE
        )
        if(NOT ${UNZIP_RETURN_CODE} EQUAL 0)
            message(FATAL_ERROR "unzip failed")
        endif()
    endif()

    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    add_subdirectory(${CMAKE_BINARY_DIR}/torchvision/vision-0.14.1 torchvision)
    include_directories(${CMAKE_BINARY_DIR}/torchvision/vision-0.14.1/)
    set(TORCHVISION_LIBRARIES torchvision)
else()
    message(WARNING "Torch is not support for the CPU architecture.")
    add_compile_definitions(NO_TORCH)
endif()


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   opentera_webrtc_ros_msgs#   sensor_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES face_cropping
#  CATKIN_DEPENDS cv_bridge opentera_webrtc_ros_msgs roscpp image_transport sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_executable(face_cropping_node
    src/FaceDetector.h
    src/OpencvFaceDetector.h
    src/OpencvFaceDetector.cpp
    src/TorchFaceDetector.h
    src/TorchFaceDetector.cpp
    src/SinglePositionKalmanFilter.h
    src/SinglePositionKalmanFilter.cpp
    src/FaceCropper.h
    src/FaceCropper.cpp
    src/face_cropping_node.cpp
)

target_link_libraries(face_cropping_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${TORCHVISION_LIBRARIES}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/face_cropping.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/face_cropping_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_face_cropping.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
