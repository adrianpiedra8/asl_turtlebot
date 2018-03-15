import numpy as np
from numpy import sin, cos
import scipy.linalg    # you may find scipy.linalg.block_diag useful
from ExtractLines import ExtractLines, normalize_line_parameters, angle_difference
from maze_sim_parameters import LineExtractionParams, NoiseParams, MapParams

class EKF(object):

    def __init__(self, x0, P0, Q):
        self.x = x0    # Gaussian belief mean
        self.P = P0    # Gaussian belief covariance
        self.Q = Q     # Gaussian control noise covariance (corresponding to dt = 1 second)

    # Updates belief state given a discrete control step (Gaussianity preserved by linearizing dynamics)
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def transition_update(self, u, dt):
        g, Gx, Gu = self.transition_model(u, dt)

        self.x = g
        self.P = np.matmul(np.matmul(Gx, self.P), np.transpose(Gx)) + dt * np.matmul(np.matmul(Gu, self.Q), np.transpose(Gu))

    # Propagates exact (nonlinear) state dynamics; also returns associated Jacobians for EKF linearization
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: (g, Gx, Gu)
    #      g  - result of belief mean self.x propagated according to the system dynamics with control u for dt seconds
    #      Gx - Jacobian of g with respect to the belief mean self.x
    #      Gu - Jacobian of g with respect to the control u
    def transition_model(self, u, dt):
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    # Updates belief state according to a given measurement (with associated uncertainty)
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def measurement_update(self, rawZ, rawR):
        z, R, H = self.measurement_model(rawZ, rawR)
        if z is None:    # don't update if measurement is invalid (e.g., no line matches for line-based EKF localization)
            return

        sigma = np.matmul(np.matmul(H, self.P), np.transpose(H)) + R
        K = np.matmul(np.matmul(self.P, np.transpose(H)), np.linalg.inv(sigma))

        self.x = self.x + np.matmul(K, z).flatten()
        self.P = self.P - np.matmul(np.matmul(K, sigma), np.transpose(K))

    # Converts raw measurement into the relevant Gaussian form (e.g., a dimensionality reduction);
    # also returns associated Jacobian for EKF linearization
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: (z, R, H)
    #       z - measurement mean (for simple measurement models this may = rawZ)
    #       R - measurement covariance (for simple measurement models this may = rawR)
    #       H - Jacobian of z with respect to the belief mean self.x
    def measurement_model(self, rawZ, rawR):
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class Localization_EKF(EKF):

    def __init__(self, x0, P0, Q, map_lines, tf_base_to_camera, g):
        self.map_lines = map_lines                    # 2xJ matrix containing (alpha, r) for each of J map lines
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Unicycle dynamics (Turtlebot 2)
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x

        # handle special case of small omega (theta is approximately constant)
        if np.abs(om) < 1e-8:
            g = np.array([
                x + dt * v * cos(th + om * dt),
                y + dt * v * sin(th + om * dt),
                th + om * dt
            ])

            Gx = np.array([
                [1, 0, -dt * v * sin(th + om * dt)],
                [0, 1, dt * v * cos(th + om * dt)],
                [0, 0, 1]
            ])

            Gu = np.array([
                [dt * cos(th + om * dt), -dt**2 * v * sin(th + om * dt) / 2],
                [dt * sin(th + om * dt), dt**2 * v * cos(th + om * dt) / 2],
                [0, dt]
            ])

        else:
            g = np.array([
                x + v / om * (sin(th + om * dt) - sin(th)),
                y - v / om * (cos(th + om * dt) - cos(th)),
                th + om * dt
                ])

            Gx = np.array([
                [1, 0, v / om * (cos(th + om * dt) - cos(th))],
                [0, 1, -v / om * (-sin(th + om * dt) + sin(th))],
                [0, 0, 1]
                ])

            Gu = np.array([
                [(sin(th + om * dt) - sin(th)) / om, (-v / om**2) * (sin(th + om * dt) - sin(th)) + (v / om) * cos(th + om * dt) * dt],
                [-(cos(th + om * dt) - cos(th)) / om, (v / om**2) * (cos(th + om * dt) - cos(th)) + (v / om) * sin(th + om * dt) * dt],
                [0, dt]
                ])

        return g, Gx, Gu

    # Given a single map line m in the world frame, outputs the line parameters in the scanner frame so it can
    # be associated with the lines extracted from the scanner measurements
    # INPUT:  m = (alpha, r)
    #       m - line parameters in the world frame
    # OUTPUT: (h, Hx)
    #       h - line parameters in the scanner (camera) frame
    #      Hx - Jacobian of h with respect to the belief mean self.x
    def map_line_to_predicted_measurement(self, m):
        alpha, r = m

        # find the pose of the robot in the world frame
        x, y, th = self.x
        # find the pose of the camera in the robot frame
        x_cam, y_cam, th_cam = self.tf_base_to_camera

        # compute the line parameters in the camera frame
        h = np.array([alpha - th - th_cam, r - x * cos(alpha) - y * sin(alpha) - x_cam * cos(alpha - th) - y_cam * sin(alpha - th)])

        # compute the Jacobian of h with respect to the belief mean
        Hx = np.array([
            [0, 0, -1],
            [-cos(alpha), -sin(alpha), -x_cam * sin(alpha - th) + y_cam * cos(alpha - th)]
            ])

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Given lines extracted from the scanner data, tries to associate to each one the closest map entry
    # measured by Mahalanobis distance
    # INPUT:  (rawZ, rawR)
    #    rawZ - 2xI matrix containing (alpha, r) for each of I lines extracted from the scanner data (in scanner frame)
    #    rawR - list of I 2x2 covariance matrices corresponding to each (alpha, r) column of rawZ
    # OUTPUT: (v_list, R_list, H_list)
    #  v_list - list of at most I innovation vectors (predicted map measurement - scanner measurement)
    #  R_list - list of len(v_list) covariance matrices of the innovation vectors (from scanner uncertainty)
    #  H_list - list of len(v_list) Jacobians of the innovation vectors with respect to the belief mean self.x
    def associate_measurements(self, rawZ, rawR):
        v_list = []
        R_list = []
        H_list = []
        # loop through each of the I lines extracted from the scanner data
        for i in range(rawZ.shape[1]):
            # initialize the arrays of v, H, and d for each line
            v = []
            H = []
            d = []
            # loop through all of the lines in the map
            for j in range(self.map_lines.shape[1]):
                # transform the line parameters from the world frame to the camera frame
                h, Hx = self.map_line_to_predicted_measurement(self.map_lines[:, j])
                # add the current Hx to the array of H for the current map line
                H.append(Hx)
                # add the current innovation to the array of v for the current lines from the map and data
                v.append(rawZ[:, i] - h)
                # compute the innovation covariance
                S = np.matmul(np.matmul(Hx, self.P), np.transpose(Hx)) + rawR[i]
                # add the current Mahalanobis distance to the array of d for the current lines from the map and data
                d.append(np.matmul(np.matmul(v[j].reshape(1,2), np.linalg.inv(S)), v[j].reshape((2,1))))

            # find the index corresponding to the minimum Mahalanobis distance
            valid_idx = np.argmin(d)
            # check that the minimum Mahalanobis distance falls into the validation gate
            if d[valid_idx] < (self.g)**2:
                # add the corresponding v to the list of v
                v_list.append(v[valid_idx])
                # add the current R to the list of R
                R_list.append(rawR[i])
                # add the corresponding H to the list of H
                H_list.append(H[valid_idx])

        return v_list, R_list, H_list

    # Assemble one joint measurement, covariance, and Jacobian from the individual values corresponding to each
    # matched line feature
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        z = np.row_stack(item for v in v_list for item in v)
        R = scipy.linalg.block_diag(*R_list)
        H = np.row_stack(np.asarray(H_list))

        return z, R, H


class SLAM_EKF(EKF):

    def __init__(self, x0, P0, Q, tf_base_to_camera, g):
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Combined Turtlebot + map dynamics
    # Adapt this method from Localization_EKF.transition_model.
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x[:3]

        #### TODO ####
        # compute g, Gx, Gu (some shape hints below)
        # g = np.copy(self.x)
        # Gx = np.eye(self.x.size)
        # Gu = np.zeros((self.x.size, 2))
        ##############

        # handle special case of small omega (theta is approximately constant)
        if np.abs(om) < 1e-8:
            g = np.array([
                x + dt * v * cos(th + om * dt),
                y + dt * v * sin(th + om * dt),
                th + om * dt
            ])

            Gx = np.array([
                [1, 0, -dt * v * sin(th + om * dt)],
                [0, 1, dt * v * cos(th + om * dt)],
                [0, 0, 1]
            ])

            Gu = np.array([
                [dt * cos(th + om * dt), -dt**2 * v * sin(th + om * dt) / 2],
                [dt * sin(th + om * dt), dt**2 * v * cos(th + om * dt) / 2],
                [0, dt]
            ])

        else:
            g = np.array([
                x + v / om * (sin(th + om * dt) - sin(th)),
                y - v / om * (cos(th + om * dt) - cos(th)),
                th + om * dt
                ])

            Gx = np.array([
                [1, 0, v / om * (cos(th + om * dt) - cos(th))],
                [0, 1, -v / om * (-sin(th + om * dt) + sin(th))],
                [0, 0, 1]
                ])

            Gu = np.array([
                [(sin(th + om * dt) - sin(th)) / om, (-v / om**2) * (sin(th + om * dt) - sin(th)) + (v / om) * cos(th + om * dt) * dt],
                [-(cos(th + om * dt) - cos(th)) / om, (v / om**2) * (cos(th + om * dt) - cos(th)) + (v / om) * sin(th + om * dt) * dt],
                [0, dt]
                ])

        g = np.append(g, self.x[3:])
        Gx = scipy.linalg.block_diag(Gx, np.eye(len(self.x[3:])))
        Gu = np.row_stack((Gu, np.zeros((len(self.x[3:]), 2))))

        return g, Gx, Gu

    # Combined Turtlebot + map measurement model
    # Adapt this method from Localization_EKF.measurement_model.
    #
    # The ingredients for this model should look very similar to those for Localization_EKF.
    # In particular, essentially the only thing that needs to change is the computation
    # of Hx in map_line_to_predicted_measurement and how that method is called in
    # associate_measurements (i.e., instead of getting world-frame line parameters from
    # self.map_lines, you must extract them from the state self.x)
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H (should be identical to Localization_EKF.measurement_model above)
        ##############

        z = np.row_stack(item for v in v_list for item in v)
        R = scipy.linalg.block_diag(*R_list)
        H = np.row_stack(np.asarray(H_list))

        return z, R, H

	# Adapt this method from Localization_EKF.map_line_to_predicted_measurement.
	#
	# Note that instead of the actual parameters m = (alpha, r) we pass in the map line index j
	# so that we know which components of the Jacobian to fill in.
	def map_line_to_predicted_measurement(self, j):
		alpha, r = self.x[(3+2*j):(3+2*j+2)]    # j is zero-indexed! (yeah yeah I know this doesn't match the pset writeup)

		#### TODO ####
		# compute h, Hx (you may find the skeleton for computing Hx below useful)

		# Hx = np.zeros((2,self.x.size))
		# Hx[:,:3] = FILLMEIN
		# First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
		# if j > 1:
		#     Hx[0, 3+2*j] = FILLMEIN
		#     Hx[1, 3+2*j] = FILLMEIN
		#     Hx[0, 3+2*j+1] = FILLMEIN
		#     Hx[1, 3+2*j+1] = FILLMEIN

		##############

		# find the pose of the robot in the world frame
		x, y, th = self.x[:3]
		# find the pose of the camera in the robot frame
		x_cam, y_cam, th_cam = self.tf_base_to_camera

		# compute the line parameters in the camera frame
		h = np.array([alpha - th - th_cam, r - x * cos(alpha) - y * sin(alpha) - x_cam * cos(alpha - th) - y_cam * sin(alpha - th)])
		
		# compute the Jacobian of h with respect to the belief mean
		Hx = np.zeros((2,self.x.size))
		Hx[:,:3] = np.array([
			[0, 0, -1],
			[-cos(alpha), -sin(alpha), -x_cam * sin(alpha - th) + y_cam * cos(alpha - th)]
			])
		# First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
		if j > 1:
			Hx[0, 3+2*j] = 1
			Hx[1, 3+2*j] = x * sin(alpha) - y * cos(alpha) + x_cam * sin(alpha - th) - y_cam * cos(alpha - th)
			Hx[0, 3+2*j+1] = 0
			Hx[1, 3+2*j+1] = 1

		flipped, h = normalize_line_parameters(h)
		if flipped:
			Hx[1,:] = -Hx[1,:]

		return h, Hx

    # Adapt this method from Localization_EKF.associate_measurements.
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        ##############

        v_list = []
        R_list = []
        H_list = []
        # loop through each of the I lines extracted from the scanner data
        for i in range(rawZ.shape[1]):
            # initialize the arrays of v, H, and d for each line
            v = []
            H = []
            d = []
            # loop through all of the lines in the state
            for j in range(len(self.x[3:]) / 2):
                # transform the line parameters from the world frame to the camera frame
                h, Hx = self.map_line_to_predicted_measurement(j)
                # add the current Hx to the array of H for the current state line
                H.append(Hx)
                # add the current innovation to the array of v for the current lines from the state and data
                v.append(rawZ[:, i] - h)
                # compute the innovation covariance
                S = np.matmul(np.matmul(Hx, self.P), np.transpose(Hx)) + rawR[i]
                # add the current Mahalanobis distance to the array of d for the current lines from the state and data
                d.append(np.matmul(np.matmul(v[j].reshape(1,2), np.linalg.inv(S)), v[j].reshape((2,1))))

            # find the index corresponding to the minimum Mahalanobis distance
            valid_idx = np.argmin(d)
            # check that the minimum Mahalanobis distance falls into the validation gate
            if d[valid_idx] < (self.g)**2:
                # add the corresponding v to the list of v
                v_list.append(v[valid_idx])
                # add the current R to the list of R
                R_list.append(rawR[i])
                # add the corresponding H to the list of H
                H_list.append(H[valid_idx])

        return v_list, R_list, H_list
