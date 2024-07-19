cmake_minimum_required(VERSION 3.16)
project(opentera_protobuf_messages)

find_package(ament_cmake REQUIRED)

find_package(Protobuf REQUIRED COMPONENTS protobuf_messages protoc)

set(PROTOBUF_INPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/opentera_messages/proto)
set(PROTOBUF_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/opentera_protobuf_messages)
set(GOOGLE_PROTO_PATH ${Protobuf_INCLUDE_DIR}/google/protobuf)
set(PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/opentera_messages/python/__init__.py
    DESTINATION ${PROTOBUF_OUTPUT_PATH}
)

FILE(GLOB_RECURSE protofiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${PROTOBUF_INPUT_PATH}/*.proto")

set(protobuf_python_targets)

foreach(file ${protofiles})
    message(STATUS "Protobuf: processing ${file}")
    get_filename_component (name_without_extension ${file} NAME_WE)
    #Add command to generate target
    set(outfile ${PROTOBUF_OUTPUT_PATH}/${name_without_extension}_pb2.py)
    add_custom_command(
        OUTPUT ${outfile}
        DEPENDS ${file}
        COMMAND ${PROTOC_EXECUTABLE} ${file} -I=opentera_messages/proto -I${GOOGLE_PROTO_PATH} --python_out ${PROTOBUF_OUTPUT_PATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
    #Add target
    set(output_target opentera_messages_${name_without_extension}.proto.target)
    list(APPEND protobuf_python_targets ${output_target})
    add_custom_target(
        ${output_target}
        DEPENDS ${outfile}
        VERBATIM
    )
endforeach(file)

add_custom_target(python-messages ALL DEPENDS ${protobuf_python_targets})

ament_python_install_package(${PROJECT_NAME})

ament_package()
