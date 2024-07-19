cmake_minimum_required(VERSION 3.8)
project(opentera_webrtc_demos)

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue/package.json
    COMMAND npm install
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue
    VERBATIM
)

add_custom_target(
    opentera_werbrtc_teleop_frontend-install ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp
    VERBATIM
)

file(
    GLOB_RECURSE
    FRONTEND_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue/src/*.ts
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue/src/*.js
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue/src/*.vue
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue/src/*.scss
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist.stamp
    DEPENDS ${FRONTEND_FILES} ${CMAKE_CURRENT_BINARY_DIR}/node_modules.stamp opentera_werbrtc_teleop_frontend-install
    COMMAND npm run build
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/dist.stamp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/opentera-webrtc-teleop-frontend/teleop-vue
    VERBATIM
)

add_custom_target(
    opentera_werbrtc_teleop_frontend-build ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dist.stamp
    VERBATIM
)

install(PROGRAMS
    scripts/robot_status.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    config
    models
    worlds
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
    opentera-webrtc-teleop-frontend/teleop-vue/dist
    DESTINATION share/${PROJECT_NAME}/opentera-webrtc-teleop-frontend/teleop-vue
)

#######################
## Required programs ##
#######################
function(assert_program_installed PROGRAM)
    find_program(prog-${PROGRAM} ${PROGRAM})
    if(${prog-${PROGRAM}} STREQUAL "prog-${PROGRAM}-NOTFOUND")
        message(FATAL_ERROR "'${PROGRAM}' needs to be installed")
    endif()
endfunction()

assert_program_installed("npm")

ament_package()
