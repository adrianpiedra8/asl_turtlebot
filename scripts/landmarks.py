#!/usr/bin/env python

import numpy as np
import rospy
from visualization_msgs.msg import Marker
import pdb

def findmatch(existing, new, thresh):
    if existing.shape[0] == 0:
        return None

    dist = np.linalg.norm(existing - new, axis=1)
    if np.any(dist < thresh):
        return np.where(dist < thresh)[0][0]
    return None

def publish_marker(name, loc, type):

    marker_pub = rospy.Publisher('/markers/' + name, Marker, queue_size=10)
    # publishes the detected object and its location
    marker = Marker()


    marker.header.frame_id = "/map"

    if type == 'stop_sign':
        marker.type = marker.SPHERE
        marker.color.r = 1
        marker.color.b = 0
        marker.color.g = 0
    elif type == 'animal':
        marker.type = marker.CUBE
        marker.color.r = 0
        marker.color.b = 1
        marker.color.g = 0
    else: 
        error('unrecognized type in publish marker')

    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = loc[0]
    marker.pose.position.y = loc[1]
    marker.pose.position.z = 0
    marker.ns = type
    marker.text = name

    marker_pub.publish(marker)

class StopSigns:
    def __init__(self, dist_thresh=1):
        self.locations = np.zeros((0, 2))
        self.observations_count = np.zeros((0))
        self.dist_thresh = dist_thresh

    def publish_all(self):
        for i in range(self.length()):
            publish_marker('stop_sign' + str(i), self.locations[i,:], 'stop_sign')

    def length(self):
        return self.locations.shape[0]

    def pprint(self):
        print('StopSigns pprint:')
        for i in range(self.locations.shape[0]):
            print("    id: {} location: [{:.2f}, {:.2f}] observation count: {}".format(
                  i, self.locations[i,0], self.locations[i,1], self.observations_count[i]))
        print("")
        
    def add_observation(self, observation):
        existing = findmatch(self.locations, observation, self.dist_thresh)
        if existing != None:
            self.update_location(existing, observation)
        else:
            self.locations = np.vstack((self.locations, observation))
            self.observations_count = np.append(self.observations_count, 1)
            index = self.locations.shape[0] - 1
            rospy.loginfo("Adding new stop sign. Index %d is now [%f, %f]",
                  index, self.locations[index, 0], self.locations[index, 1])

    def update_location(self, index, observation):
        prev = self.locations[index]
        n = self.observations_count[index]

        # Average of all observations we have seen so far
        self.locations[index] = (prev*n + observation)/(n+1)

        # Increment observations count
        self.observations_count[index] = n + 1

        # rospy.loginfo("Incorporating stop sign observation. Index %f incorporated [%f, %f] and is now [%f, %f]",
              # index, observation[0], observation[1], self.locations[index, 0], self.locations[index, 1])

class AnimalWaypoints:
    def __init__(self, dist_thresh=1):
        self.poses = np.zeros((0, 3))
        self.locations = np.zeros((0,2))
        self.dist_thresh = dist_thresh
        self.bbox_heights = np.zeros((0))
        self.observations_count = np.zeros((0))
        self.animal_types = []
        self.animal_theta_g = []


    def cull(self, min_observations):
        new_poses = np.zeros((0, 3))
        new_locations = np.zeros((0,2))
        new_bbox_heights = np.zeros((0))
        new_observations_count = np.zeros((0))
        new_animal_types = []

        for i in range(self.length()):
            if self.observations_count[i] >= min_observations:
                new_locations = np.vstack((new_locations, self.locations[i,:]))
                new_poses = np.vstack((new_poses, self.poses[i,:]))
                new_bbox_heights = np.append(new_bbox_heights, self.bbox_heights[i])
                new_observations_count = np.append(new_observations_count, self.observations_count[i])
                new_animal_types.append(self.animal_types[i])

        self.poses = new_poses
        self.locations = new_locations
        self.bbox_heights = new_bbox_heights
        self.observations_count = new_observations_count
        self.animal_types = new_animal_types

    def reorder(self, index):
        print("Reordering waypoints to {}".format(index))
        self.poses = self.poses[index,:]
        self.locations = self.locations[index,:]
        self.bbox_heights = self.bbox_heights[index]
        self.observations_count = self.observations_count[index]
        self.animal_types =  [self.animal_types[i] for i in index.tolist()]
 
    def publish_all(self):
        for i in range(self.length()):
            publish_marker('animal' + str(i), self.locations[i,:], 'animal')

    def length(self):
        return self.poses.shape[0]
        
    def pprint(self):
        print('AnimalWaypoints pprint:')
        for i in range(self.locations.shape[0]):
            print("    id: {} location: [{:.2f}, {:.2f}] observation count: {} height: {} pose: [{:.2f}, {:.2f}, {:.2f}] type: {}".format(
                  i, self.locations[i,0], self.locations[i,1], self.observations_count[i], 
                  self.bbox_heights[i], self.poses[i,0], self.poses[i,1], self.poses[i,2], 
                  self.animal_types[i]))
        print("")

    def add_observation(self, observation, pose, bbox_height, animal_type, theta_goal):
        existing = findmatch(self.locations, observation, self.dist_thresh)
        if existing != None:
            self.update_location(existing, observation, pose, bbox_height)
            return existing
        else:
            self.locations = np.vstack((self.locations, observation))
            self.poses = np.vstack((self.poses, pose))
            self.bbox_heights = np.append(self.bbox_heights, bbox_height)
            self.observations_count = np.append(self.observations_count, 1)
            self.animal_types.append(animal_type)
            self.animal_theta_g.append(theta_goal)

            index = self.locations.shape[0] - 1
            rospy.loginfo("Adding new animal waypoint. Index: %d, location: [%f, %f], pose: [%f, %f, %f], bbox_height: %f, animal type: %s",
                  index, self.locations[index, 0], self.locations[index, 1], 
                  self.poses[index, 0], self.poses[index, 1], self.poses[index, 2], 
                  self.bbox_heights[index], self.animal_types[index])
            return index

    def update_location(self, index, observation, pose, bbox_height):
        prev = self.locations[index]
        n = self.observations_count[index]

        # Average of all observations we have seen so far
        self.locations[index] = (prev*n + observation)/(n+1)

        # Increment observations count
        self.observations_count[index] = n + 1

        rospy.loginfo("Incorporating animal waypoint observation. Index %d now has %d observations",
                      index, self.observations_count[index])

        # Only update pose if new observation has a bigger bounding box
        if bbox_height > self.bbox_heights[index]:
            self.poses[index] = pose
            self.bbox_heights[index] = bbox_height
            rospy.loginfo("Updated animal waypoint pose %f to [%f, %f, %f] ", 
                          index, self.poses[index, 0], self.poses[index, 1], self.poses[index, 2])


    def pop(self):
        if len(self.bbox_heights) > 0:
            waypoint = self.poses[0,:]
            animal_type = self.animal_types[0]
            self.poses = np.delete(self.poses, 0, axis=0)
            self.locations = np.delete(self.locations, 0, axis=0)
            self.bbox_heights = np.delete(self.bbox_heights, 0)
            del self.animal_types[0]
            return waypoint, animal_type
        else:
            return None, None

class ExploreWaypoints:
    def __init__(self):
        self.poses = np.zeros((0, 3))
 
    def length(self):
        return self.poses.shape[0]
        
    def add_exploration_waypoint(self, pose):
        self.poses = np.vstack((self.poses, pose))

    def pop(self):
        if self.length() > 0:
            waypoint = self.poses[0,:]
            self.poses = np.delete(self.poses, 0, axis=0)
            return waypoint
        else:
            return None
