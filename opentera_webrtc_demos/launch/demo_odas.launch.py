from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():

    opentera_webrtc_demos_launch_dir = (
        Path(get_package_share_directory("opentera_webrtc_demos")) / "launch"
    )
    odas_ros_launch_dir = Path(get_package_share_directory("odas_ros")) / "launch"

    odas_rviz_config_file = str(
        Path(get_package_share_directory("odas_ros"))
        / "config"
        / "rviz"
        / "odas_rviz.rviz"
    )

    odas_configuration_path_folder = (
        Path(get_package_share_directory("opentera_webrtc_demos")) / "config"
    )

    return LaunchDescription(
        [
            # Arguments Demo
            DeclareLaunchArgument(
                "signaling_server_hostname", default_value="localhost"
            ),
            DeclareLaunchArgument("signaling_server_port", default_value="8080"),
            DeclareLaunchArgument("signaling_server_password", default_value="abc"),
            DeclareLaunchArgument("centered_robot", default_value="true"),
            DeclareLaunchArgument("robot_vertical_offset", default_value="180"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("gazebo_gui", default_value="false"),
            DeclareLaunchArgument("is_stand_alone", default_value="true"),
            DeclareLaunchArgument("camera_id", default_value="0"),
            DeclareLaunchArgument("use_outgoing_face_cropping", default_value="false"),
            DeclareLaunchArgument("use_incoming_face_cropping", default_value="false"),
            # Arguments ODAS
            DeclareLaunchArgument("frame_id", default_value="odas"),
            DeclareLaunchArgument("visualization", default_value="true"),
            DeclareLaunchArgument("odas_rviz", default_value="false"),
            DeclareLaunchArgument(
                "odas_rviz_cfg",
                default_value=f"-d {odas_rviz_config_file}",
            ),
            DeclareLaunchArgument("local", default_value="true"),
            DeclareLaunchArgument("use_echo_cancellation", default_value="true"),
            DeclareLaunchArgument(
                "echo_cancellation_source",
                default_value="alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input",
            ),
            DeclareLaunchArgument("echo_cancellation_sink", default_value="__default"),
            DeclareLaunchArgument(
                "echo_cancellation_dest", default_value="odas_echo_cancelled"
            ),
            DeclareLaunchArgument("ec_volume_percent", default_value="100"),
            DeclareLaunchArgument(
                "configuration_path",
                default_value=str(
                    odas_configuration_path_folder
                    / "demo_respeaker_usb_4_mic_array_ec.cfg"
                ),
                condition=IfCondition(LaunchConfiguration("use_echo_cancellation")),
            ),
            DeclareLaunchArgument(
                "configuration_path",
                default_value=str(
                    odas_configuration_path_folder
                    / "demo_respeaker_usb_4_mic_array.cfg"
                ),
                condition=UnlessCondition(LaunchConfiguration("use_echo_cancellation")),
            ),
            # Simulation time for whole launch context
            SetParameter(name='use_sim_time', value=True),
            # Demo
            IncludeLaunchDescription(
                XMLLaunchDescriptionSource(
                    str(opentera_webrtc_demos_launch_dir / "demo.launch.xml")
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
                    "is_stand_alone": LaunchConfiguration("is_stand_alone"),
                    "camera_id": LaunchConfiguration("camera_id"),
                    "use_outgoing_face_cropping": LaunchConfiguration(
                        "use_outgoing_face_cropping"
                    ),
                    "use_incoming_face_cropping": LaunchConfiguration(
                        "use_incoming_face_cropping"
                    ),
                }.items(),
            ),
            # Sound interface echo cancellation
            Node(
                package="odas_ros",
                executable="echocancel.sh",
                name="echocancel",
                arguments=[
                    LaunchConfiguration("echo_cancellation_source"),
                    LaunchConfiguration("echo_cancellation_sink"),
                    LaunchConfiguration("echo_cancellation_dest"),
                    LaunchConfiguration("ec_volume_percent"),
                ],
                condition=IfCondition(LaunchConfiguration("use_echo_cancellation")),
            ),
            # ODAS
            IncludeLaunchDescription(
                XMLLaunchDescriptionSource(
                    str(odas_ros_launch_dir / "odas.launch.xml")
                ),
                launch_arguments={
                    "configuration_path": LaunchConfiguration("configuration_path"),
                    "frame_id": LaunchConfiguration("frame_id"),
                    "visualization": LaunchConfiguration("visualization"),
                    "rviz": LaunchConfiguration("odas_rviz"),
                    "rviz_cfg": LaunchConfiguration("odas_rviz_cfg"),
                    "local": LaunchConfiguration("local"),
                    "force_publish_tf": "true",
                    "use_echo_cancellation": LaunchConfiguration(
                        "use_echo_cancellation"
                    ),
                    "echo_cancelled_signal_topic": "/odas/sss",
                }.items(),
            ),
        ]
    )
