# opentera_webrtc_robot_gui

## Description

The GUI is written with Qt 5.12+ and displays stream information (ROS images) from the WebRTC connection. It will utimately be used to display internal robot state, control the microphone and camera usage and enable calling from the robot's side. At the moment, calls are initiated from the web with [opentera-webrtc-teleop-frontend](https://github.com/introlab/opentera-webrtc-teleop-frontend) and sessions are automatically joined.

> The GUI is unfinished at the moment but usable to display basic video streams.

## Customization 

In order to adjust the gui to different screen sizes, aspect ratios and resolutions, a series of properties can be changed from a JSON file. By default this json file is named deviceProperties and can be found in the resources folder. It can however by overriden by using the ros parameter `device_properties_path`. The following properties are available.

### width and height

The desired resolution of the window, by default 600px by 1024px.

### diagonal length

The diagonal length of the window on the user's screen. This will adjust the size of certain UI elements, in order to be visible on smaller and bigger screens, by default 7 inches.

### defaultLocalCameraWidth and defaultLocalCameraHeight

During a session, the local camera shrinks down to a small window that's draggable and resizable. These parameters state the camera window's default size and have default values of 320px by 240px. These match the aspect ratio of the opentera-webrtc-ros demo video stream, but it's recommended that these be changed to match the aspect ratio of whatever is the incoming video, in order to remove black bars.

### defaultLocalCameraOpacity

The local camera window's default opacity, it can also be changed from the configuration menu. By default has an value of 90%.

### defaultLocalCameraX and defaultLocalCameraY

The default position of the camera window in X and Y coordinates. Noting that the top left is the (0, 0) point. These values can be negative wich will place the window from the opposite side. These properties respectively have default values of 10px and -10px, wich places the window 10 pixels from the borders at the bottom left.






