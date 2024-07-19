# face_cropping

## Face Detection Models

- [haarcascade_frontalface_default.xml](models/haarcascade_frontalface_default.xml) Haar feature-based cascade classifiers to detect faces. The model is from the OpenCV repository and the license is at the beginning of the file.
- [lbpcascade_frontalface_improved.xml](models/lbpcascade_frontalface_improved.xml) LBP feature-based cascade classifiers to detect faces. The model is from the OpenCV repository and the license is at the beginning of the file.
- [small_yunet_*.pt*](models/small_yunet_*.pt) The models are derived from [YuNet](https://github.com/ShiqiYu/libfacedetection), but it is faster since it is optimized for big faces. Their backbones are trained on ImageNet using knowledge distillation. The face detectors are trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) without the small faces. The training code is available in the [dnn_training](dnn_training/) folder. SimOTA is used for the face assignation. The [SimOTA code](dnn_training/mmdet/) is from (MMDetection)[https://github.com/open-mmlab/mmdetection].

## LibTorch
LibTorch and torchvision are required to use small_yunet_* models. LibTorch and torchvision are automatically downloaded on AMD64 computers.
On Jetson computer, you need to install [PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) built by NVIDIA and compile [torchvision for C++](https://github.com/pytorch/vision#using-the-models-on-c).

To use the GPU, you need to set the option FACE_CROPPER_USE_CUDA to ON.

## Resource usage
The CPU and memory usage are measured on an AMD RYZEN 7 3700X at 30 Hz.
The APs are measured on a subset of WIDER FACE. The subset includes faces occupying at least 10% of the width and the height of the image.

| Model                | LibTorch Required | CPU (%, 100% = 1 core) | Memory (MB) | AP@0.25 | AP@0.50 | AP@0.75 |
| -------------------- | ----------------- | ---------------------- | ----------- | ------- | ------- | ------- |
| haarcascade          | No                | 220.4                  | 299.7       | 0.5973  | 0.5649  | 0.0314  |
| lbpcascade           | No                | 154.7                  | 299.1       | 0.4141  | 0.4014  | 0.0705  |
| small_yunet_0.25_160 | Yes               | 11.4                   | 273.9       | 0.6896  | 0.4940  | 0.1298  |
| small_yunet_0.25_320 | Yes               | 16.7                   | 276.4       | 0.7969  | 0.7019  | 0.2669  |
| small_yunet_0.25_640 | Yes               | 34.3                   | 297.4       | 0.8266  | 0.7601  | 0.4237  |
| small_yunet_0.5_160  | Yes               | 12.7                   | 274.4       | 0.7930  | 0.6834  | 0.3183  |
| small_yunet_0.5_320  | Yes               | 20.2                   | 282.2       | 0.8622  | 0.8034  | 0.4840  |
| small_yunet_0.5_640  | Yes               | 50.1                   | 315.8       | 0.8780  | 0.8466  | 0.6018  |


## `face_cropping_node`

Implements a ROS node that subscribes to an image stream. It detects the biggest face in the received frame and crops it out.

It can be enabled and disabled from the robot GUI by pressing the face cropping button, by default it is enabled.

### Parameters

- `face_detection_model` (string): The face detection model (`haarcascade`, `lbpcascade`, `small_yunet_0.25_160`,
`small_yunet_0.25_320`, `small_yunet_0.25_640`, `small_yunet_0.5_160`, `small_yunet_0.5_320` or `small_yunet_0.5_640`).
- `use_gpu_if_available` (bool): Indicates whether to use the GPU or not.
- `min_face_width` (double): The minimum face width.
- `min_face_height` (double): The minimum face height.
- `output_width` (int): The output face image width.
- `output_height` (int): The output face image height.
- `adjust_brightness` (bool): Indicates whether to adjust the brightness or not.

### Subscribed Topics

- `enable_face_cropping` [std_msgs/Bool](http://docs.ros2.org/foxy/api/std_msgs/msg/Bool.html): The topic to enable or disable the node.
- `input_image` [sensor_msgs/Image](http://docs.ros2.org/foxy/api/sensor_msgs/msg/Image.html): The input image topic.

### Published Topics

- `output_image` [sensor_msgs/Image](http://docs.ros2.org/foxy/api/sensor_msgs/msg/Image.html): The output image topic.
