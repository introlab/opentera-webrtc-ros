#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

#Qt 
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap

# ROS
import rospy
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
        image = QImage(frame.data, frame.width, frame.height, QImage.Format_RGB888).rgbSwapped()
        self._label.setPixmap(QPixmap.fromImage(image))


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self._peer_image_subscriber = rospy.Subscriber('/webrtc_image', PeerImage, self._on_peer_image, queue_size=10)
        self._image_view = ImageView(self)
        self.setCentralWidget(self._image_view)

    def _on_peer_image(self, image: PeerImage):
        print('image from', image.sender.id)
        self._image_view.setImage(image.frame)

    

if __name__ == '__main__':
    # Init ROS
    rospy.init_node('opentera_webrtc_robot_gui', anonymous=True)
    

    # Init Qt App
    # Create an instance of QApplication
    app = QApplication(sys.argv)

    # Create Main Window
    window = MainWindow()
    window.showMaximized()

    # Execute application
    sys.exit(app.exec_())


