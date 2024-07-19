cmake_minimum_required(VERSION 3.8)
project(opentera_webrtc_ros_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(audio_utils REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/DatabaseEvent.msg"
    "msg/DeviceEvent.msg"
    "msg/JoinSessionEvent.msg"
    "msg/JoinSessionReplyEvent.msg"
    "msg/LabelArray.msg"
    "msg/LabelEdit.msg"
    "msg/Label.msg"
    "msg/LabelSimple.msg"
    "msg/LabelSimpleArray.msg"
    "msg/LabelSimpleEdit.msg"
    "msg/LeaveSessionEvent.msg"
    "msg/LogEvent.msg"
    "msg/ParticipantEvent.msg"
    "msg/StopSessionEvent.msg"
    "msg/UserEvent.msg"
    "msg/OpenTeraEvent.msg"
    "msg/Peer.msg"
    "msg/PeerData.msg"
    "msg/PeerImage.msg"
    "msg/PeerAudio.msg"
    "msg/PeerStatus.msg"
    "msg/RobotStatus.msg"
    "msg/Waypoint.msg"
    "msg/WaypointArray.msg"

    "srv/ChangeMapView.srv"
    "srv/ImageGoalToMapGoal.srv"
    "srv/SetString.srv"

    DEPENDENCIES std_msgs geometry_msgs sensor_msgs audio_utils
)

ament_package()
