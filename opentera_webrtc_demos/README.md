# opentera-webrtc-demos

To launch the stand alone demo try :

```bash
# Load ROS environment
$ source ~/teleop_ws/install/setup.bash
# Needed for simulation
$ export TURTLEBOT3_MODEL=waffle
# Run the demo
$ ros2 launch opentera_webrtc_demos demo.launch is_stand_alone:=true
```

To launch the stand alone demo with ODAS functionnality, you will need the [ReSpeaker 4-mic array](https://respeaker.io/usb_4_mic_array/) plugged-in your computer:
```bash
# Load ROS environment
$ source ~/teleop_ws/install/setup.bash
# Needed for simulation
$ export TURTLEBOT3_MODEL=waffle
# Run the demo
$ ros2 launch opentera_webrtc_demos demo_odas.launch is_stand_alone:=true
```

Once launched go to the following URL :
>[http://localhost:8080/index.html#/user?name=dev&pwd=abc&robot=BEAM](http://localhost:8080/index.html#/user?name=dev&pwd=abc&robot=BEAM)
