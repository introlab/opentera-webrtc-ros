cmake_minimum_required(VERSION 3.0.2)
project(opentera_webrtc_ros_msgs)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    message_generation
    sensor_msgs
    audio_utils
)

add_message_files(
    FILES
        DatabaseEvent.msg
        DeviceEvent.msg
        JoinSessionEvent.msg
        JoinSessionReplyEvent.msg
        LabelArray.msg
        LabelEdit.msg
        Label.msg
        LabelSimple.msg
        LabelSimpleArray.msg
        LabelSimpleEdit.msg
        LeaveSessionEvent.msg
        LogEvent.msg
        ParticipantEvent.msg
        StopSessionEvent.msg
        UserEvent.msg
        OpenTeraEvent.msg
        Peer.msg
        PeerData.msg
        PeerImage.msg
        PeerAudio.msg
        PeerStatus.msg
        RobotStatus.msg
        Waypoint.msg
        WaypointArray.msg
)

add_service_files(
    FILES
        SetString.srv
)

generate_messages(
    DEPENDENCIES
        std_msgs
        sensor_msgs
        audio_utils
        geometry_msgs
)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs message_runtime
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)
