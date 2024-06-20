#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

# Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal

# ROS
import rclpy
import rclpy.node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from opentera_webrtc_ros_msgs.msg import PeerImage
from threading import Thread

import sys


class AsyncSpinner:
    def __init__(self) -> None:
        self._spinner = None
    
    def start_spin(self, *nodes: rclpy.node.Node):
        self._spinner = Thread(target=rclpy.spin, args=(nodes,))
        self._spinner.start()

    def stop(self):
        if rclpy.ok():
            rclpy.shutdown()
        if self._spinner:
            self._spinner.join()


class ImageView(QWidget):
    setImage = pyqtSignal(Image)
    
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self._layout = QVBoxLayout(self)
        self._label = QLabel(self)
        self._layout.addWidget(self._label)
        self._label.setText('Hello!')

        self.setImage.connect(self._setImage)

    def _setImage(self, frame: Image):
        width = frame.width
        height = frame.height
        encoding = frame.encoding
        image = QImage(frame.data, frame.width, frame.height,
                       QImage.Format_RGB888).rgbSwapped()
        self._label.setPixmap(QPixmap.fromImage(image))



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        self.node = rclpy.node.Node('opentera_webrtc_robot_gui')

        self._peer_image_subscriber = self.node.create_subscription(
            PeerImage, '/webrtc_image', self._on_peer_image, 10)
        self._image_view = ImageView(self)
        self.setCentralWidget(self._image_view)

    def _on_peer_image(self, image: PeerImage):
        print('image from', image.sender.id)
        self._image_view.setImage.emit(image.frame)


if __name__ == '__main__':
    # Init ROS
    rclpy.init()

    # Init Qt App
    # Create an instance of QApplication
    app = QApplication(sys.argv)

    # Create Async Spinner
    spinner = AsyncSpinner()

    # Create Main Window
    window = MainWindow()
    window.showMaximized()

    # Execute application
    spinner.start_spin(window.node)
    ret = app.exec_()

    # Stop spinner
    spinner.stop()

    sys.exit(ret)
