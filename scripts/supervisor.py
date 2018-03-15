#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String, Bool, Int8
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject, TSalesRequest, TSalesCircuit
import landmarks
import tf
import math
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
# from asl_turtlebot import finalcount.wav
from enum import Enum
import numpy as np
import traveling_salesman
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

BIKE_STOP_TIME = 5

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    STOP = 1
    CROSS = 2
    NAV = 3
    PLAN_RESCUE = 4
    REQUEST_RESCUE = 5
    GO_TO_ANIMAL = 6
    RESCUE_ANIMAL = 7
    BIKE_STOP = 8
    VICTORY = 9

class Supervisor:
    """ the state machine of the turtlebot """

    def __init__(self):
        rospy.init_node('turtlebot_supervisor', anonymous=True)

        self.soundhandle = SoundClient()

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
        self.playsound = True

        # string for target animal
        self.target_animal = None

        # status flag for traveling salesman circuit received
        self.tsales_circuit_received = 1

        # lock waypoints
        self.lock_animal_waypoints = 0

        # current mode
        self.mode = Mode.IDLE
        self.modeafterstop = Mode.IDLE
        self.modeafterhonk = Mode.IDLE
        self.last_mode_printed = None

        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rescue_ready_publisher = rospy.Publisher('/ready_to_rescue', Bool, queue_size=10)
        self.tsales_request_publisher = rospy.Publisher('/tsales_request', TSalesRequest, queue_size=10)

        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        rospy.Subscriber('/detector/bird', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/cat', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/dog', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/horse', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/sheep', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/cow', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/elephant', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/bear', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/zebra', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/giraffe', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/bicycle', DetectedObject, self.bicycle_detected_callback)
        rospy.Subscriber('/rescue_on', Bool, self.rescue_on_callback)
        rospy.Subscriber('/cmd_state', Int8, self.cmd_state_callback)
        rospy.Subscriber('/tsales_circuit', TSalesCircuit, self.tsales_circuit_callback)

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
        return any(dist < STOP_MIN_DIST for dist in dist2stop)

    def animal_detected_callback(self, msg):
        """ callback for when the detector has found an animal """

        # Check the location is valid i.e. 3rd element is non-zero
        if msg.location_W[2] == 1.0:
            pose = np.array([self.x, self.y, self.theta])
            bbox_height = msg.corners[3] - msg.corners[1]

            observation = msg.location_W[:2]

            animal_type = msg.name

            # only add animals in the exploration states
            if not self.lock_animal_waypoints:
                self.animal_waypoints.add_observation(observation, pose, bbox_height, animal_type)
                self.theta_g = msg.location_W[3]

    def bicycle_detected_callback(self, msg):
    	"""callback for when the detector has found a bicycle"""
        self.honk = True
        self.bike_detected_start = rospy.get_rostime()

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

    def init_plan_rescue(self):
        print('init plan rescue')
        self.tsales_circuit_received = 0
        self.lock_animal_waypoints = 1

        if self.animal_waypoints.poses.shape[0] > 0:
            tsales_request = TSalesRequest()
            tsales_request.goal_x = self.animal_waypoints.poses[:,0].tolist()
            tsales_request.goal_y = self.animal_waypoints.poses[:,1].tolist()
            tsales_request.do_fast = 0
            print('publish tsales request')
            self.tsales_request_publisher.publish(tsales_request) 
        else: 
            self.tsales_circuit_received = 1

    def tsales_circuit_callback(self, msg): 
        print('tsales circuit callback')
        try:
            circuit = np.array(map(int, msg.circuit))
        except:
            rospy.loginfo('Traveling salesman failed')
            self.tsales_circuit_received = 1
            return

        if circuit.shape[0] == self.animal_waypoints.poses.shape[0]:
            self.animal_waypoints.reorder(circuit)
        else:         
            rospy.loginfo('Traveling salesman failed')

        self.tsales_circuit_received = 1

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
            if self.honk:

                if (self.playsound):
                    self.soundhandle.playWave('/home/aa274/catkin_ws/src/asl_turtlebot/BICYCLE.wav', 1.0)
                    self.playsound = False
                    print("Playing the sound")

            if (rospy.get_rostime() - self.bike_detected_start > rospy.Duration.from_sec(BIKE_STOP_TIME)):
                self.honk = False
                self.playsound = True
                print("I'm stopping the honking")
                self.mode = self.modeafterhonk
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

            if self.honk:
                self.mode = Mode.BIKE_STOP
                self.modeafterhonk = Mode.CROSS

        elif self.mode == Mode.NAV:
            self.lock_animal_waypoints = 0

            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                if self.stop_check(): # Returns True if within STOP_MIN_DIST
                    self.mode = Mode.STOP
                    self.modeafterstop = Mode.NAV
                else:
                    self.nav_to_pose()

            if self.honk:
                self.mode = Mode.BIKE_STOP
                self.modeafterhonk = Mode.NAV
            
        elif self.mode == Mode.PLAN_RESCUE:
            if not self.init_flag:
                self.init_plan_rescue()
                self.init_flag = 1
            
            if self.tsales_circuit_received:
                self.mode = Mode.REQUEST_RESCUE

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
                    self.mode = Mode.VICTORY

                self.init_go_to_animal()

            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.RESCUE_ANIMAL
            else:
                if self.stop_check(): # Returns True if within STOP_MIN_DIST
                    self.mode = Mode.STOP
                    self.modeafterstop = Mode.GO_TO_ANIMAL
                else:
                    self.nav_to_pose()

            if self.honk:
                self.mode = Mode.BIKE_STOP
                self.modeafterhonk = Mode.GO_TO_ANIMAL

        elif self.mode == Mode.RESCUE_ANIMAL:
            if not self.init_flag:
                self.init_flag = 1
                self.init_rescue_animal()

            if self.has_rescued():
                rospy.loginfo("Rescued a: %s", self.target_animal)
                if self.animal_waypoints.length() > 0:
                    self.mode = Mode.GO_TO_ANIMAL
                else:
                    self.mode = Mode.VICTORY

        elif self.mode == Mode.VICTORY:
            # self.stay_idle()
            twist = Twist()
            twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
            twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0.1
            self.cmd_vel_publisher.publish(twist)

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
