import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    return LaunchDescription([
        
        # Turtlebot base position arguments
        DeclareLaunchArgument('x_pos', default_value='1.0', description='Turtlebot X position'),
        DeclareLaunchArgument('y_pos', default_value='0.5', description='Turtlebot Y position'),
        DeclareLaunchArgument('z_pos', default_value='0.0', description='Turtlebot Z position'),

        # RTAB-Map config arguments
        DeclareLaunchArgument('use_rtabmap', default_value='true', description='Whether to use RTAB-Map'),
        DeclareLaunchArgument('open_rviz', default_value='false', description='Whether to open RViz'),
        DeclareLaunchArgument('rtabmap_viz', default_value='false', description='Whether to visualize RTAB-Map'),
        DeclareLaunchArgument('gazebo_gui', default_value='false', description='Whether to use Gazebo GUI'),
        DeclareLaunchArgument('use_nav2', default_value='true', description='Whether to use navigation2'),
        DeclareLaunchArgument('with_camera', default_value='true', description='Whether to use camera'),
        DeclareLaunchArgument('localization', default_value='false', description='Whether to use localization'),
        DeclareLaunchArgument('database_path', default_value='~/.ros/rtabmap.db', description='RTAB-Map database path'),
        DeclareLaunchArgument('pause', default_value='false', description='Pause Gazebo'),
        DeclareLaunchArgument('rtabmap_args', default_value='', description='Arguments to pass to RTAB-Map'),  # <-- Declare the missing argument


        # Gazebo Launch (ExecuteProcess instead of Node)
        ExecuteProcess(
            cmd=[
                'gzserver', '--verbose',
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so',
                PathJoinSubstitution([FindPackageShare('opentera_webrtc_demos'), 'worlds', 'turtlebot3_house.world']),
                '--pause' if LaunchConfiguration('pause') == 'true' else ''
            ],
            output='screen',
            condition=UnlessCondition(LaunchConfiguration('gazebo_gui'))
        ),

        # Robot description and urdf
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_urdf',
            arguments=['-topic', '/robot_description', '-entity', 'turtlebot3_waffle', '-x', LaunchConfiguration('x_pos'),
                       '-y', LaunchConfiguration('y_pos'), '-z', LaunchConfiguration('z_pos')]
        ),
        
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': Command(
                ['xacro ', PathJoinSubstitution([FindPackageShare('turtlebot3_beam_description'), 'urdf', 'turtlebot3_waffle.urdf.xacro'])])
            }],
            remappings=[('/robot_description', '/robot_description')]
        ),

        # RTAB-Map (with namespace)
        Node(
            package='rtabmap_sync',
            executable='rgbd_sync',
            name='rgbd_sync',
            namespace='rtabmap',
            remappings=[
                ('rgb/image', '/r200/rgb/image_raw'),
                ('depth/image', '/r200/depth/image_raw'),
                ('rgb/camera_info', '/r200/rgb/camera_info')
            ],
            condition=IfCondition(LaunchConfiguration('use_rtabmap'))
        ),

        # Node to handle RTAB-Map SLAM
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            namespace='rtabmap',
            arguments=[LaunchConfiguration('rtabmap_args')],
            parameters=[
                {'database_path': LaunchConfiguration('database_path')},
                {'frame_id': 'base_footprint'},
                {'subscribe_rgb': False},
                {'subscribe_depth': False},
                {'subscribe_rgbd': True},
                {'subscribe_scan': True},
                {'approx_sync': True},
                {'map_always_update': True},
                {'map_empty_ray_tracing': True},
                {'use_action_for_goal': True},
                {'Reg/Strategy': '1'},
                {'Reg/Force3DoF': 'true'},
                {'GridGlobal/MinSize': '20'},
                # Set the 'Mem_IncrementalMemory' parameter based on localization argument
                {'Mem/IncrementalMemory': "False"}
            ],
            remappings=[
                ('scan', '/scan'),
                ('odom', '/odom'),
                ('rgbd_image', 'rgbd_image'),
                ('map', '/map')
            ],
            condition=IfCondition(LaunchConfiguration('use_rtabmap'))
        ),

        # Visualization with rtabmap_viz
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            namespace='rtabmap',
            arguments=[PathJoinSubstitution([FindPackageShare('rtabmap_demos'), 'launch', 'config', 'rgbd_gui.ini'])],
            parameters=[
                {'subscribe_scan': True},
                {'subscribe_odom': True},
                {'frame_id': 'base_footprint'},
                {'approx_sync': True}
            ],
            remappings=[
                ('odom', '/odom'),
                ('scan', '/scan')
            ],
            condition=IfCondition(LaunchConfiguration('rtabmap_viz'))
        ),

        # Navigation2 (optional)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'])),
            launch_arguments={'use_sim_time': 'true'}.items(),
            condition=IfCondition(LaunchConfiguration('use_nav2'))
        ),

        # RViz (optional)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=[PathJoinSubstitution([FindPackageShare('turtlebot3_navigation2'), 'rviz', 'tb3_navigation2.rviz'])],
            condition=IfCondition(LaunchConfiguration('open_rviz'))
        ),

        # Goal Manager Node
        Node(
            package='opentera_webrtc_ros',
            executable='goal_manager.py',
            name='goal_manager',
            remappings=[('waypoint_reached', 'webrtc_data_outgoing')]
        ),

        # Labels Manager Node
        Node(
            package='opentera_webrtc_ros',
            executable='labels_manager.py',
            name='labels_manager',
            remappings=[
                ('waypoint_reached', 'webrtc_data_outgoing'),
                ('stored_labels_text', 'webrtc_data_outgoing')
            ]
        )
    ])
