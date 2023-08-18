#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from opentera_webrtc_ros_msgs.msg import PeerImage


IMAGE_DELAY = 300


class PeerImageMock:
    def __init__(self):
        self._image_sub = rospy.Subscriber('image', Image, self._image_cb, queue_size=10)
        self._peer_image_pub = rospy.Publisher('peer_image', PeerImage, queue_size=10)
        self._delayed_images = []

    def _image_cb(self, msg):
        self._publish_peer_image('id_0', msg)

        self._delayed_images.append(msg)
        if len(self._delayed_images) >= IMAGE_DELAY:
            self._publish_peer_image('id_1', self._delayed_images[0])
            self._delayed_images = self._delayed_images[1:]

    def _publish_peer_image(self, id, image):
        msg = PeerImage()
        msg.sender.id = id
        msg.frame = image
        self._peer_image_pub.publish(msg)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('peer_image_mock')
    peer_image_mock = PeerImageMock()
    peer_image_mock.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
