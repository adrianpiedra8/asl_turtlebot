#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
from asl_turtlebot.msg import DetectedObject

class Tester:

    def __init__(self):
        rospy.init_node('tester', anonymous=True)


    def rescue_on(self):
        rescue_on_publisher = rospy.Publisher('/rescue_on', Bool, queue_size=10)
    	while (1):
        	rescue_on_publisher.publish(True)



    def object_msg(self):
        object_publisher = rospy.Publisher('/detector/none', DetectedObject, queue_size=10)
        while 1:
            # publishes the detected object and its location
            object_msg = DetectedObject()
            object_msg.id = 0
            object_msg.name = 'None'
            object_msg.confidence = 0
            object_msg.distance = 0
            object_msg.thetaleft = 0
            object_msg.thetaright = 0
            object_msg.corners = [0,0,0,0]
            object_msg.location_W = [0,0,0,0] #pos_obj_W

            object_publisher.publish(object_msg)

if __name__ == '__main__':
    s = Tester()
    # s.rescue_on()
    s.object_msg()