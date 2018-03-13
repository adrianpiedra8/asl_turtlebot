#!/usr/bin/env python

import numpy as np

def findmatch(existing, new, thresh):
    dist = np.linalg.norm(existing - new)
    if np.any(dist < thresh):
        return np.where(dist < thresh)[0][0]
    return None

class StopSigns:
    def __init__(self, dist_thresh=1):
        self.locations = np.zeros((0, 2))
        self.observations_count = np.zeros((0))
        self.dist_thresh = dist_thresh

    def add_observation(self, observation):
        existing = findmatch(self.locations, observation)
        if existing != None:
            self.update_location(existing, observation)
        else:
            self.locations = np.vstack((self.locations, observation))
            self.observations_count.append(1)

    def update_location(self, index, observation):
        prev = self.locations[index]
        n = self.observations_count[index]

        # Average of all observations we have seen so far
        self.locations[index] = (prev*n + observation)/(n+1)

        # Increment observations count
        self.observations_count[index] = n + 1

class AnimalWaypoints:
    def __init__(self, dist_thresh=1):
        self.poses = np.zeros((0, 3))
        self.locations = np.zeros((0,2))
        self.dist_thresh = dist_thresh
        self.bbox_heights = np.zeros((0))

    def add_observation(self, observation, pose, bbox_height):
        existing = findmatch(self.locations, observation)
        if existing != None:
            self.update_location(existing, observation, pose, bbox_height)
        else:
            self.locations = np.vstack((self.locations, observation))
            self.poses = np.vstack((self.poses, pose))
            self.bbox_heights = np.append(self.bbox_heights, bbox_height)

    def update_location(self, index, observation, pose, bbox_height):
        # Only update if new observation has a bigger bounding box
        if bbox_height > self.bbox_heights[index]:
            self.locations[index] = observation
            self.poses[index] = pose
            self.bbox_heights[index] = bbox_height

    def pop(self):
        if len(self.bbox_heights) > 0:
            waypoint = self.poses[0,:]
            self.poses = np.delete(self.poses, 0, axis=0)
            self.locations = np.delete(self.locations, 0, axis=0)
            self.bbox_heights = np.delete(self.bbox_heights, 0)
            return waypoint
        else:
            return None
