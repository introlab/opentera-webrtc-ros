#!/usr/bin/env python
import cv2

import rospy
from cv_bridge import CvBridge
from opentera_webrtc_ros_msgs.msg import PeerImage


class PeerImageDisplay:
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._image_sub = rospy.Subscriber('peer_image', PeerImage, self._peer_image_cb, queue_size=10)

    def _peer_image_cb(self, msg):
        cv2.imshow(msg.sender.id, self._cv_bridge.imgmsg_to_cv2(msg.frame, 'bgr8'))
        cv2.waitKey(1)

    def _publish_peer_image(self, id, image):
        msg = PeerImage()
        msg.sender.id = id
        msg.frame = image

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('peer_image_display')
    peer_image_display = PeerImageDisplay()
    peer_image_display.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
