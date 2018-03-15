#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String, Bool, Int8
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject
import landmarks
import tf
import math
from enum import Enum
import numpy as np
import pdb

# threshold at which we consider the robot at a location
POS_EPS = .1
THETA_EPS = .3

# time to stop at a stop sign
STOP_TIME = 3

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = .3

# minimum distance frmo a stop sign to leave CROSS mode
CROSS_MIN_DIST = .4

# time taken to cross an intersection
CROSSING_TIME = 3

# time taken to rescue an animal
RESCUE_TIME = 3

# Distance threshold to consider to stop detections as
# the same stop sign
STOP_SIGN_DIST_THRESH = 0.6

# Distance threshold to consider to animal detections as
# the same stop sign
ANIMAL_DIST_THRESH = 0.6


# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 1
    STOP = 2
    CROSS = 3
    NAV = 4
    EXPLORE = 5
    REQUEST_RESCUE = 6
    GO_TO_ANIMAL = 7
    RESCUE_ANIMAL = 8
    BIKE_STOP = 9

class Supervisor:
    """ the state machine of the turtlebot """

    def __init__(self):
        rospy.init_node('turtlebot_supervisor', anonymous=True)

        # current pose
        self.x = 0
        self.y = 0
        self.theta = 0

        # pose goal
        self.x_g = 0
        self.y_g = 0
        self.theta_g = 0

        # init flag used for running init function for each state
        self.init_flag = 0

        # Landmark lists
        self.stop_signs = landmarks.StopSigns(dist_thresh=STOP_SIGN_DIST_THRESH) # List of coordinates for all stop signs
        self.animal_waypoints = landmarks.AnimalWaypoints(dist_thresh=ANIMAL_DIST_THRESH)

        # flag that determines if the rescue can be initiated
        self.rescue_on = False

        # flag that determines if the robot has found a bicycle and should honk
        #self.bicycles = []
        self.honk = False

        # string for target animal
        self.target_animal = None

        # current mode
        self.mode = Mode.IDLE
        self.modeafterstop = Mode.IDLE
        self.last_mode_printed = None

        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rescue_ready_publisher = rospy.Publisher('/ready_to_rescue', Bool, queue_size=10)

        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber('/detector/dog', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/cat', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/bicycle', DetectedObject, self.bicycle_detected_callback)
        rospy.Subscriber('/rescue_on', Bool, self.rescue_on_callback)
        rospy.Subscriber('/cmd_state', Int8, self.cmd_state_callback)

        self.trans_listener = tf.TransformListener()

    def cmd_state_callback(self, msg):
        self.mode = Mode(msg.data)

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """

        self.x_g = msg.pose.position.x
        self.y_g = msg.pose.position.y
        rotation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(rotation)
        self.theta_g = euler[2]

        self.mode = Mode.NAV

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # Check the location is valid i.e. 3rd element is non-zero
        if msg.location_W[2] == 1.0:
            observation = msg.location_W[:2]
            self.stop_signs.add_observation(observation)

    def stop_check(self):
        """ checks if within stopping threshold """
        current_pose = [self.x, self.y]
        dist2stop = []
        for i in range(self.stop_signs.locations.shape[0]):
            dist2stop.append(np.linalg.norm(current_pose - self.stop_signs.locations[i,:])) # Creates list of distances to all stop signs
        return (self.mode == Mode.NAV and any(dist < STOP_MIN_DIST for dist in dist2stop))

    def animal_detected_callback(self, msg):
        """ callback for when the detector has found an animal """

        # Check the location is valid i.e. 3rd element is non-zero
        if msg.location_W[2] == 1.0:
            pose = np.array([self.x, self.y, self.theta])
            bbox_height = msg.corners[3] - msg.corners[1]

            observation = msg.location_W[:2]

            animal_type = msg.name

            # only add animals in the exploration state
            if self.mode == Mode.EXPLORE:
                self.animal_waypoints.add_observation(observation, pose, bbox_height, animal_type)

    def bicycle_detected_callback(self, msg):
    	"""callback for when the detector has found a bicycle"""
            self.honk = True
            self.bike_detected_start = rospy.get_rostime()
            self.mode = Mode.BIKE_STOP
            #self.stop_signs.add_observation(observation)

    def rescue_on_callback(self, msg):
        """callback for when the rescue is ready"""
        self.rescue_on = msg.data

    def nav_to_pose(self):
        """ sends the current desired pose to the navigator """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self,x,y,theta):
        """ checks if the robot is at a pose within some threshold """

        return (abs(x-self.x)<POS_EPS and abs(y-self.y)<POS_EPS and abs(theta-self.theta)<THETA_EPS)

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return (self.mode == Mode.STOP and (rospy.get_rostime()-self.stop_sign_start)>rospy.Duration.from_sec(STOP_TIME))

    def has_crossed(self):
        """ checks if crossing maneuver is over """
        current_pose = [self.x, self.y]
        dist2stop = []
        for i in range(self.stop_signs.locations.shape[0]):
            dist2stop.append(np.linalg.norm(current_pose - self.stop_signs.locations[i,:])) # Creates list of distances to all stop signs
        return (self.mode == Mode.CROSS and all(dist > CROSS_MIN_DIST for dist in dist2stop)) # (rospy.get_rostime()-self.cross_start)>rospy.Duration.from_sec(CROSSING_TIME))

    def init_go_to_animal(self):
        # remove the animal from the rescue queue
        waypoint, animal_type = self.animal_waypoints.pop()
        print waypoint, animal_type

        if np.any(waypoint == None):
            pass
        else:
            self.x_g = waypoint[0]
            self.y_g = waypoint[1]
            self.theta_g = waypoint[2]
            self.target_animal = animal_type

    def init_rescue_animal(self):
        """ initiates an animal rescue """

        self.rescue_start = rospy.get_rostime()
        self.mode = Mode.RESCUE_ANIMAL

    def has_rescued(self):
        """checks if animal has been rescued"""

        return (self.mode == Mode.RESCUE_ANIMAL and (rospy.get_rostime()-self.rescue_start)>rospy.Duration.from_sec(RESCUE_TIME))

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        try:
            (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.x = translation[0]
            self.y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            self.theta = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        #self.bicycles.publish_all()
        self.stop_signs.publish_all()
        self.animal_waypoints.publish_all()

        if self.honk:
        	### Make it honk
        	print("I'm honking!!!!!!")

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode
            self.init_flag = 0

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.IDLE:
            # send zero velocity
            self.stay_idle()

        elif self.mode == Mode.BIKE_STOP:

        	if(rospy.get_rostime - self.bike_detected_start > 5):
        		self.honk == False
        		self.mode = Mode.NAV
        	else:
        		self.stay_idle()

        elif self.mode == Mode.STOP:
            # at a stop sign

            if not self.init_flag:
                self.init_flag = 1
                self.init_stop_sign()

            if self.has_stopped():
                self.mode = Mode.CROSS
            else:
                self.stay_idle()

        elif self.mode == Mode.CROSS:
            # crossing an intersection

            if self.has_crossed():
                self.mode = self.modeafterstop
            else:
                self.nav_to_pose()

        elif self.mode == Mode.NAV:
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                if self.stop_check(): # Returns True if within STOP_MIN_DIST
                    self.mode = Mode.STOP
                    self.modeafterstop = Mode.NAV
                else:
                    self.nav_to_pose()

        elif self.mode == Mode.EXPLORE:
            # explore with teleop
            pass

        elif self.mode == Mode.REQUEST_RESCUE:
            # publish message that rescue is ready
            rescue_ready_msg = True
            self.rescue_ready_publisher.publish(rescue_ready_msg)

            # when rescue on message is received, tranisition to rescue
            if self.rescue_on:
                if self.animal_waypoints.length() > 0:
                    self.mode = Mode.GO_TO_ANIMAL
                else:
                    self.mode = Mode.IDLE

        elif self.mode == Mode.GO_TO_ANIMAL:
            # navigate to the animal

            if not self.init_flag:
                self.init_flag = 1
                if self.animal_waypoints.length() == 0:
                    self.mode = Mode.IDLE

                self.init_go_to_animal()


            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.RESCUE_ANIMAL
            else:
                if self.stop_check(): # Returns True if within STOP_MIN_DIST
                    self.mode = Mode.STOP
                    self.modeafterstop = Mode.GO_TO_ANIMAL
                else:
                    self.nav_to_pose()

        elif self.mode == Mode.RESCUE_ANIMAL:
            if not self.init_flag:
                self.init_flag = 1
                self.init_rescue_animal()

            if self.has_rescued():
                rospy.loginfo("Rescued a: %s", self.target_animal)
                if self.animal_waypoints.length() > 0:
                    self.mode = Mode.GO_TO_ANIMAL
                else:
                    self.mode = Mode.IDLE

        else:
            raise Exception('This mode is not supported: %s'
                % str(self.mode))

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
