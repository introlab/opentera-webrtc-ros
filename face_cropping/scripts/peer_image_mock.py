#!/usr/bin/env python3
import rclpy
import rclpy.node
from sensor_msgs.msg import Image
from opentera_webrtc_ros_msgs.msg import PeerImage


IMAGE_DELAY = 300


class PeerImageMock(rclpy.node.Node):
    def __init__(self):
        super().__init__('peer_image_mock')

        self._image_sub = self.create_subscription(Image, 'image', self._image_cb, 10)
        self._peer_image_pub = self.create_publisher(PeerImage, 'peer_image', 10)
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
        rclpy.spin(self)


def main():
    rclpy.init()
    peer_image_mock = PeerImageMock()
    
    try:
        peer_image_mock.run()
    except KeyboardInterrupt:
        pass

    peer_image_mock.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
