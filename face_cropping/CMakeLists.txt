cmake_minimum_required(VERSION 3.16)
project(face_cropping)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(opentera_webrtc_ros_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ament_index_cpp REQUIRED)

find_package(OpenCV REQUIRED)

find_package(OpenMP QUIET)
if (OpenMP_FOUND)
    set(OpenMP_LIBRARIES OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP is not found.")
    add_compile_definitions(NO_TORCH)
endif()

option(FACE_CROPPER_USE_CUDA "Add the support of CUDA" OFF)

# Jetson platforms use libtorch installed from the NVIDIA forum
if (EXISTS "/etc/nv_tegra_release")
    execute_process(COMMAND python3 -c "import torch;" RESULT_VARIABLE IMPORT_TORCH_EXIT_CODE OUTPUT_QUIET)

    if (${IMPORT_TORCH_EXIT_CODE} EQUAL 0)
        message(STATUS "PyTorch is found.")

        execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
                OUTPUT_VARIABLE CMAKE_PREFIX_PATH)

        find_package(Torch REQUIRED)
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
        find_package(TorchVision QUIET)
        if (TorchVision_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
            set(TORCHVISION_LIBRARIES TorchVision::TorchVision)
            include_directories(SYSTEM ${TorchVision_INCLUDE_DIR})
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

    add_subdirectory(${CMAKE_BINARY_DIR}/torchvision/vision-0.14.1 torchvision SYSTEM)
    include_directories(SYSTEM ${CMAKE_BINARY_DIR}/torchvision/vision-0.14.1/)
    set(TORCHVISION_LIBRARIES torchvision)
    add_compile_definitions(TORCHVISION_CSRC_INCLUDE)

    file(GLOB_RECURSE TORCH_SHARED_OBJECTS "${CMAKE_BINARY_DIR}/libtorch/libtorch/lib/lib*.so*")

    install(FILES ${TORCH_SHARED_OBJECTS} DESTINATION lib)
    install(TARGETS torchvision DESTINATION lib)
else()
    message(WARNING "Torch is not supported for the CPU architecture.")
    add_compile_definitions(NO_TORCH)
endif()

set(COMMON_FILES src/FaceDetector.h
    src/FaceDetector.cpp
    src/OpencvFaceDetector.h
    src/OpencvFaceDetector.cpp
    src/TorchFaceDetector.h
    src/TorchFaceDetector.cpp
    src/SinglePositionKalmanFilter.h
    src/SinglePositionKalmanFilter.cpp
    src/FaceCropper.h
    src/FaceCropper.cpp
    src/FaceCroppingNodeConfiguration.h
    src/FaceCroppingNodeConfiguration.cpp
    src/OpencvUtils.h
)

add_executable(face_cropping_node
    ${COMMON_FILES}
    src/face_cropping_node.cpp
)
ament_target_dependencies(face_cropping_node rclcpp sensor_msgs opentera_webrtc_ros_msgs cv_bridge image_transport ament_index_cpp)
target_link_libraries(face_cropping_node
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${TORCHVISION_LIBRARIES}
    ${OpenMP_LIBRARIES}
)
target_include_directories(
    face_cropping_node
    SYSTEM
    PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
)
install(TARGETS face_cropping_node DESTINATION lib/${PROJECT_NAME})

add_executable(face_cropping_webrtc_node
    ${COMMON_FILES}
    src/face_cropping_webrtc_node.cpp
)
ament_target_dependencies(face_cropping_webrtc_node rclcpp sensor_msgs opentera_webrtc_ros_msgs cv_bridge image_transport ament_index_cpp)
target_link_libraries(face_cropping_webrtc_node
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${TORCHVISION_LIBRARIES}
    ${OpenMP_LIBRARIES}
)
target_include_directories(
    face_cropping_webrtc_node
    SYSTEM
    PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
)
install(TARGETS face_cropping_webrtc_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

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

install(PROGRAMS
    scripts/peer_image_display.py
    scripts/peer_image_mock.py
    DESTINATION lib/${PROJECT_NAME}
)

# Launch files
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
