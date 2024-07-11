from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():

    opentera_webrtc_demos_launch_dir = (
        Path(get_package_share_directory("opentera_webrtc_demos")) / "launch"
    )
    opentera_client_ros_launch_dir = (
        Path(get_package_share_directory("opentera_client_ros")) / "launch"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "signaling_server_hostname", default_value="localhost"
            ),
            DeclareLaunchArgument("signaling_server_port", default_value="8080"),
            DeclareLaunchArgument("signaling_server_password", default_value="abc"),
            DeclareLaunchArgument("centered_robot", default_value="true"),
            DeclareLaunchArgument("robot_vertical_offset", default_value="180"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("gazebo_gui", default_value="false"),
            DeclareLaunchArgument("camera_id", default_value="0"),
            DeclareLaunchArgument("use_outgoing_face_cropping", default_value="false"),
            DeclareLaunchArgument("use_incoming_face_cropping", default_value="false"),
            # Simulation time for whole launch context
            SetParameter(name='use_sim_time', value=True),
            # Nodes
            IncludeLaunchDescription(
                XMLLaunchDescriptionSource(
                    str(opentera_client_ros_launch_dir / "client.launch.xml")
                ),
                launch_arguments={
                    "opentera_client_config_file": "~/.ros/opentera/client_config.json",
                }.items(),
            ),
            IncludeLaunchDescription(
                XMLLaunchDescriptionSource(
                    str(opentera_webrtc_demos_launch_dir / "opentera_demo.launch.xml")
                ),
                launch_arguments={
                    "signaling_server_hostname": LaunchConfiguration(
                        "signaling_server_hostname"
                    ),
                    "signaling_server_port": LaunchConfiguration(
                        "signaling_server_port"
                    ),
                    "signaling_server_password": LaunchConfiguration(
                        "signaling_server_password"
                    ),
                    "centered_robot": LaunchConfiguration("centered_robot"),
                    "robot_vertical_offset": LaunchConfiguration(
                        "robot_vertical_offset"
                    ),
                    "rviz": LaunchConfiguration("rviz"),
                    "gazebo_gui": LaunchConfiguration("gazebo_gui"),
                    "is_stand_alone": "false",
                    "camera_id": LaunchConfiguration("camera_id"),
                    "use_outgoing_face_cropping": LaunchConfiguration(
                        "use_outgoing_face_cropping"
                    ),
                    "use_incoming_face_cropping": LaunchConfiguration(
                        "use_incoming_face_cropping"
                    ),
                }.items(),
            ),
        ]
    )
