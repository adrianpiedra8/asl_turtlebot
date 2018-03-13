import landmarks
import numpy as np
import pdb

STOP_SIGN_DIST_THRESH = 1
ANIMAL_DIST_THRESH = 1

stop_signs = landmarks.StopSigns(dist_thresh=STOP_SIGN_DIST_THRESH)
animal_waypoints = landmarks.AnimalWaypoints(dist_thresh=ANIMAL_DIST_THRESH)

stop_sign_observations = np.array([
	                               [2.5, 2.3],
	                               [-3.2, 1.4],
	                               [2.7, 2.5],
	                               [3.1, 2.2]
	                               ])
stop_signs.pprint()
for i in range(stop_sign_observations.shape[0]):
	stop_signs.add_observation(stop_sign_observations[i,:])
	stop_signs.pprint()



waypoint_observations = np.array([
	                               [2.5, 2.3],
	                               [-3.2, 1.4],
	                               [2.7, 2.5],
	                               [3.1, 2.2]
	                               ])

poses = np.array([
	                               [2.5, 2.3, 1.2],
	                               [-3.2, 1.4, 1.5],
	                               [2.7, 2.5, 1.1],
	                               [3.1, 2.2, 0.4]
	                               ])

bbox_heights = np.array([23.2, 15.3, 25.3, 26.1   ])

animal_types = np.array([1.0, 2.0, 1.0, 1.0   ])

animal_waypoints.pprint()
for i in range(stop_sign_observations.shape[0]):
	animal_waypoints.add_observation(waypoint_observations[i,:], 
		                             poses[i,:], 
		                             bbox_heights[i],
		                             animal_types[i])
	animal_waypoints.pprint()


