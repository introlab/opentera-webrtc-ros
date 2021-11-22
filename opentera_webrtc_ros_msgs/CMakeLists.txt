cmake_minimum_required(VERSION 3.0.2)
project(opentera_webrtc_ros_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
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

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   audio_utils
)

catkin_package(
   CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
