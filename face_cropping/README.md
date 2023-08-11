# face_cropping

## Face Detection Models

- [haarcascade_frontalface_default.xml](models/haarcascade_frontalface_default.xml) Haar feature-based cascade classifiers to detect faces. The model is from the OpenCV repository and the license is at the beginning of the file.
- [lbpcascade_frontalface_improved.xml](models/lbpcascade_frontalface_improved.xml) LBP feature-based cascade classifiers to detect faces. The model is from the OpenCV repository and the license is at the beginning of the file.
- [small_yunet_*.pt*](models/small_yunet_*.pt) The models are derived from [YuNet](https://github.com/ShiqiYu/libfacedetection), but it is faster since it is optimized for big faces. Their backone are trained on ImageNet using knowledge distillation. The face detector are trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) without the small faces. The training code is available in the [dnn_training](dnn_training/) folder. SimOTA is used for the face assignation. The [SimOTA code](dnn_training/mmdet/) is from (MMDetection)[https://github.com/open-mmlab/mmdetection].

## Resource usage
| ---------------------------- | ------------------- | ----------- | --------------------------- |
| Model                        | CPU (100% = 1 core) | Memory (MB) | Quality (1 worse, 5 best)   |
| ---------------------------- | ------------------- | ----------- | --------------------------- |
| haarcascade                  |                     |             |                             |
| lbpcascade                   |                     |             |                             |
| small_yunet_0.25_160         |                     |             |                             |
| small_yunet_0.25_320         |                     |             |                             |
| small_yunet_0.25_640         |                     |             |                             |
| small_yunet_0.5_160          |                     |             |                             |
| small_yunet_0.5_320          |                     |             |                             |
| small_yunet_0.5_640          |                     |             |                             |
| small_yunet_1.0_160          |                     |             |                             |
| small_yunet_1.0_320          |                     |             |                             |
| small_yunet_1.0_640          |                     |             |                             |


## `face_cropping_node`

Implements a ROS node that subscribes to an image stream. It detects the bigest face in the received frame and crops it out.

It can be enabled and disabled from the robot gui by pressing the face cropping button, by default it is enabled.

### Parameters

- `face_detection_model` (string): The face detection model (`haarcascade`, `lbpcascade`, ... or  ).
- `use_gpu_if_available` (bool): Indicates whether to use the GPU or not.
- `min_face_width` (double): The minimum face width.
- `min_face_height` (double): The minimum face height.
- `output_width` (int): The output face image width.
- `output_height` (int): The output face image height.
- `adjust_brightness` (bool): Indicates whether to adjust the brightness or not.

### Subscribed Topics

- `enable_face_cropping` [std_msgs/Bool](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Bool.html): The topic to enable or disable the node.
- `input_image` [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html): The input image topic.

### Published Topics

- `output_image` [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html): The output image topic.
