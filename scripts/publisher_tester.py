#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
from asl_turtlebot.msg import DetectedObject
from visualization_msgs.msg import Marker
import numpy as np

class Tester:

    def __init__(self):
        rospy.init_node('tester', anonymous=True)


    def rescue_on(self):
        rescue_on_publisher = rospy.Publisher('/rescue_on', Bool, queue_size=10)
        rate = rospy.Rate(1) # 1 Hz
        while (1):
            print('publishing rescue on bool')
            rescue_on_publisher.publish(True)
            rate.sleep()


    def object_msg(self):
        object_publisher0 = rospy.Publisher('/detector/cat', DetectedObject, queue_size=10)
        object_publisher1 = rospy.Publisher('/detector/dog', DetectedObject, queue_size=10)
        object_publisher2 = rospy.Publisher('/detector/stop_sign', DetectedObject, queue_size=10)
        object_publisher3 = rospy.Publisher('/detector/bicycle', DetectedObject, queue_size=10)


        publishers = [object_publisher0, object_publisher1, object_publisher2, object_publisher3]
        names = ['cat', 'dog', 'stop_sign', 'bicycle']

        rate = rospy.Rate(1) # 1 Hz
        count = 0

        while 1:
            i = count % 4

            # publishes the detected object and its location
            object_msg = DetectedObject()
            object_msg.id = 0
            object_msg.name = names[i]
            object_msg.confidence = 0
            object_msg.distance = 0
            object_msg.thetaleft = 0
            object_msg.thetaright = 0
            object_msg.corners = [0,0,0,0]
            loc = np.random.rand(2)*3
            object_msg.location_W = loc.tolist() + [1]

            print('publishing DetectedObject ' + names[i])
            publishers[i].publish(object_msg)
            count += 1
            rate.sleep()

    def spot(self):
        marker_pub = rospy.Publisher('/spot2', Marker, queue_size=10)
        rate = rospy.Rate(1) # 1 Hz
        while 1:
            # publishes the detected object and its location
            marker = Marker()

            loc = np.random.rand(2)*3


            marker.header.frame_id = "/map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = loc[0]
            marker.pose.position.y = loc[1]
            marker.pose.position.z = 0

            print('publishing Marker')
            marker_pub.publish(marker)
            rate.sleep()

if __name__ == '__main__':
    s = Tester()
    # s.rescue_on()
    s.object_msg()
    # s.spot()
