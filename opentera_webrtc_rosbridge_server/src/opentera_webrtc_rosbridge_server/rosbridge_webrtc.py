from rosbridge_library.rosbridge_protocol import RosbridgeProtocol
import uuid
import rospy
from std_msgs.msg import String
from opentera_webrtc_ros_msgs.msg import PeerData


class RosbridgeNode:
    def __init__(self) -> None:

        self.client_id_seed = uuid.uuid4()
        self.parameters = {
            "fragment_timeout": 600,
            "delay_between_messages": 0,
            "max_message_size": None,
            "unregister_timeout": 10.0,
            "bson_only_mode": False,
        }

        self.protocol = RosbridgeProtocol(
            self.client_id_seed, parameters=self.parameters
        )

        self.protocol.outgoing = self.outgoing

        self.data_in_sub = rospy.Subscriber(
            "data_in", PeerData, self.on_incoming, queue_size=10
        )
        self.data_out_pub = rospy.Publisher(
            "data_out", String, queue_size=10
        )

    def on_incoming(self, msg):
        self.protocol.incoming(msg.data)

    def outgoing(self, message):
        self.data_out_pub.publish(String(data=message))

    def run(self):
        rospy.spin()


def main():
    rospy.init_node("rosbridge_webrtc")
    try:
        node = RosbridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
