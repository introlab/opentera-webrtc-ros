import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    # Get the share directory of the opentera_webrtc_demos package
    opentera_webrtc_demos_share = get_package_share_directory('opentera_webrtc_demos')

    return LaunchDescription([
        DeclareLaunchArgument('signaling_server_hostname', default_value='localhost'),
        DeclareLaunchArgument('signaling_server_port', default_value='8080'),
        DeclareLaunchArgument('signaling_server_password', default_value='abc'),
        DeclareLaunchArgument('centered_robot', default_value='true'),
        DeclareLaunchArgument('robot_vertical_offset', default_value='180'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('gazebo_gui', default_value='false'),
        DeclareLaunchArgument('rtabmap_viz', default_value='false'),
        DeclareLaunchArgument('is_stand_alone', default_value='true'),
        DeclareLaunchArgument('camera_id', default_value='0'),
        DeclareLaunchArgument('use_outgoing_face_cropping', default_value='false'),
        DeclareLaunchArgument('use_incoming_face_cropping', default_value='false'),
        DeclareLaunchArgument('force_gstreamer_video_hardware_acceleration', default_value='true'),
        DeclareLaunchArgument('client_config_file', default_value=' '),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('certificate', default_value='/opt/ros/BIRDSEYE_WS/certificate/cert.pem'),
        DeclareLaunchArgument('key', default_value='/opt/ros/BIRDSEYE_WS/certificate/key.pem'),
        DeclareLaunchArgument('use_tls', default_value='true'),
        # Signaling server (using XML file)
        IncludeLaunchDescription(
            launch_description_source=os.path.join(opentera_webrtc_demos_share, 'launch', 'opentera_signaling_server.launch.xml'),
            launch_arguments={
                'port': LaunchConfiguration('signaling_server_port'),
                'password': LaunchConfiguration('signaling_server_password'),
                'certificate': LaunchConfiguration('certificate'),
                'key': LaunchConfiguration('key'),
                'use_tls': LaunchConfiguration('use_tls')
            }.items()
        ),

        # OpenTera Demo
        IncludeLaunchDescription(
            launch_description_source=os.path.join('/opt/ros/BIRDSEYE_WS' , 'opentera_demo.launch.xml'),
            launch_arguments={
                'signaling_server_hostname': LaunchConfiguration('signaling_server_hostname'),
                'signaling_server_port': LaunchConfiguration('signaling_server_port'),
                'signaling_server_password': LaunchConfiguration('signaling_server_password'),
                'centered_robot': LaunchConfiguration('centered_robot'),
                'robot_vertical_offset': LaunchConfiguration('robot_vertical_offset'),
                'rviz': LaunchConfiguration('rviz'),
                'gazebo_gui': LaunchConfiguration('gazebo_gui'),
                'rtabmap_viz': LaunchConfiguration('rtabmap_viz'),
                'is_stand_alone': LaunchConfiguration('is_stand_alone'),
                'camera_id': LaunchConfiguration('camera_id'),
                'use_outgoing_face_cropping': LaunchConfiguration('use_outgoing_face_cropping'),
                'use_incoming_face_cropping': LaunchConfiguration('use_incoming_face_cropping'),
                'force_gstreamer_video_hardware_acceleration': LaunchConfiguration('force_gstreamer_video_hardware_acceleration')
            }.items()
        ),
    ])
