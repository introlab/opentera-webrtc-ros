#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

# Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap

# ROS
import rclpy
import rclpy.node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from opentera_webrtc_ros_msgs.msg import PeerImage

import sys


class ImageView(QWidget):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self._layout = QVBoxLayout(self)
        self._label = QLabel(self)
        self._layout.addWidget(self._label)
        self._label.setText('Hello!')

    def setImage(self, frame: Image):
        width = frame.width
        height = frame.height
        encoding = frame.encoding
        image = QImage(frame.data, frame.width, frame.height,
                       QImage.Format_RGB888).rgbSwapped()
        self._label.setPixmap(QPixmap.fromImage(image))


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        self._node = rclpy.node.Node('opentera_webrtc_robot_gui')

        self._peer_image_subscriber = self._node.create_subscription(
            PeerImage, '/webrtc_image', self._on_peer_image, 10)
        self._image_view = ImageView(self)
        self.setCentralWidget(self._image_view)

    def _on_peer_image(self, image: PeerImage):
        print('image from', image.sender.id)
        self._image_view.setImage(image.frame)


if __name__ == '__main__':
    # Init ROS
    rclpy.init()

    # Init Qt App
    # Create an instance of QApplication
    app = QApplication(sys.argv)

    # Create Main Window
    window = MainWindow()
    window.showMaximized()

    # Execute application
    sys.exit(app.exec_())
