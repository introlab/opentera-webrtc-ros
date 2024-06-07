cmake_minimum_required(VERSION 3.8)
project(opentera_client_ros)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
    scripts/opentera_client_ros.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    config
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
