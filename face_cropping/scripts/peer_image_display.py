#!/usr/bin/env python3
import cv2

import rclpy
import rclpy.node
from cv_bridge import CvBridge
from opentera_webrtc_ros_msgs.msg import PeerImage


class PeerImageDisplay(rclpy.node.Node):
    def __init__(self):
        super().__init__('peer_image_display')

        self._cv_bridge = CvBridge()
        self._image_sub = self.create_subscription(PeerImage, 'peer_image', self._peer_image_cb, 10)

    def _peer_image_cb(self, msg):
        cv2.imshow(msg.sender.id, self._cv_bridge.imgmsg_to_cv2(msg.frame, 'bgr8'))
        cv2.waitKey(1)

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    peer_image_display = PeerImageDisplay()
    
    try:
        peer_image_display.run()
    except KeyboardInterrupt:
        pass

    peer_image_display.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
