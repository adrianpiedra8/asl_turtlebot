#!/usr/bin/env python

import rospy
import os
# from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
# watch out on the order for the next two imports lol
import tf
import tensorflow 
import numpy as np
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, LaserScan
from asl_turtlebot.msg import DetectedObject
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
import tf2_ros
import pdb

# path to the trained conv net
PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../tfmodels/ssd_mobilenet_v1_coco.pb')
PATH_TO_LABELS = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../tfmodels/coco_labels.txt')

# set to True to use tensorflow and a conv net
# False will use a very simple color thresholding to detect stop signs only
USE_TF = True
# minimum score for positive detection
MIN_SCORE = .5

def load_object_labels(filename):
    """ loads the coco object readable name """

    fo = open(filename,'r')
    lines = fo.readlines()
    fo.close()
    object_labels = {}
    for l in lines:
        object_id = int(l.split(':')[0])
        label = l.split(':')[1][1:].replace('\n','').replace('-','_').replace(' ','_')
        object_labels[object_id] = label

    return object_labels

class Detector:

    def __init__(self):
        rospy.init_node('turtlebot_detector', anonymous=True)

        # rate = rospy.Rate(1)
        # while rospy.Time.now() == rospy.Time(0):
        #     rate.sleep()

        self.bridge = CvBridge()

        if USE_TF:
            self.detection_graph = tensorflow.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tensorflow.GraphDef()
                with tensorflow.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tensorflow.import_graph_def(od_graph_def,name='')
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tensorflow.Session(graph=self.detection_graph)

        # camera and laser parameters that get updated
        self.cx = 0.
        self.cy = 0.
        self.fx = 1.
        self.fy = 1.
        self.laser_ranges = []
        self.laser_angle_increment = 0.01 # this gets updated

        self.object_publishers = {}
        self.object_labels = load_object_labels(PATH_TO_LABELS)

        self.tf_listener = tf.TransformListener()

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        rate = rospy.Rate(10) # 10 Hz
        while True:
            print("Looping")
            try:
                # notably camera_link and not camera_depth_frame below, not sure why
                raw_b2c = self.tfBuffer.lookup_transform('camera', 'base_footprint', rospy.Time()).transform
                print("Breaking loop.")
                break
            except:# tf2_ros.LookupException:
                print("Sleeping.")
                rate.sleep()

        b2c_translation = raw_b2c.translation
        b2c_rotation = [raw_b2c.rotation.x, raw_b2c.rotation.y, raw_b2c.rotation.z, raw_b2c.rotation.w]

        euler = tf.transformations.euler_from_quaternion(b2c_rotation)
        b2c_tf_theta = euler[2]    

        # Hard code theta because euler[2] doesn't seem to work for 3D rotation
        b2c_tf_theta = -1.570796326794897

        # Flip x and y because of the way we chose our camera frame
        self.base_to_camera = [b2c_translation.y,
                               b2c_translation.x,
                               b2c_tf_theta]


        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.compressed_camera_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        # rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        print("Finished Init!")

    def run_detection(self, img):
        """ runs a detection method in a given image """

        image_np = self.load_image_into_numpy_array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        if USE_TF:
            # uses MobileNet to detect objects in images
            # this works well in the real world, but requires
            # good computational resources
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes,self.d_scores,self.d_classes,self.num_d],
                feed_dict={self.image_tensor: image_np_expanded})

            return self.filter(boxes[0], scores[0], classes[0], num[0])

        else:
            # uses a simple color threshold to detect stop signs
            # this will not work in the real world, but works well in Gazebo
            # with only stop signs in the environment
            R = image_np[:,:,0].astype(np.int) > image_np[:,:,1].astype(np.int) + image_np[:,:,2].astype(np.int)
            Ry, Rx, = np.where(R)
            if len(Ry)>0 and len(Rx)>0:
                xmin, xmax = Rx.min(), Rx.max()
                ymin, ymax = Ry.min(), Ry.max()
                boxes = [[float(ymin)/image_np.shape[1], float(xmin)/image_np.shape[0], float(ymax)/image_np.shape[1], float(xmax)/image_np.shape[0]]]
                scores = [.99]
                classes = [13]
                num = 1
            else:
                boxes = []
                scores = 0
                classes = 0
                num = 0

            return boxes, scores, classes, num

    def filter(self, boxes, scores, classes, num):
        """ removes any detected object below MIN_SCORE confidence """

        f_scores, f_boxes, f_classes = [], [], []
        f_num = 0

        for i in range(num):
            if scores[i] >= MIN_SCORE:
                f_scores.append(scores[i])
                f_boxes.append(boxes[i])
                f_classes.append(int(classes[i]))
                f_num += 1
            else:
                break

        return f_boxes, f_scores, f_classes, f_num

    def load_image_into_numpy_array(self, img):
        """ converts opencv image into a numpy array """

        (im_height, im_width, im_chan) = img.shape

        return np.array(img.data).reshape((im_height, im_width, 3)).astype(np.uint8)

    def project_pixel_to_ray(self,u,v):
        """ takes in a pixel coordinate (u,v) and returns a tuple (x,y,z)
        that is a unit vector in the direction of the pixel, in the camera frame.
        This function access self.fx, self.fy, self.cx and self.cy """

        x = (u - self.cx)/self.fx
        y = (v - self.cy)/self.fy
        z = 1

        x /= np.linalg.norm(np.array([x, y, z]))
        y /= np.linalg.norm(np.array([x, y, z]))
        z /= np.linalg.norm(np.array([x, y, z]))

        return (x,y,z)

    def estimate_distance_from_thetas(self, thetaleft, thetaright, ranges):
        """ estimates the distance of an object in between two angles
        using lidar measurements """

        leftray_indx = min(max(0,int(thetaleft/self.laser_angle_increment)),len(ranges))
        rightray_indx = min(max(0,int(thetaright/self.laser_angle_increment)),len(ranges))

        if leftray_indx<rightray_indx:
            meas = ranges[rightray_indx:] + ranges[:leftray_indx]
        else:
            meas = ranges[rightray_indx:leftray_indx]

        num_m, dist = 0, 0
        for m in meas:
            if m>0 and m<float('Inf'):
                dist += m
                num_m += 1
        if num_m>0:
            dist /= num_m

        return dist

    def estimate_distance_from_image(self, box_height, obj_height, est_dist_flag):
        
        if est_dist_flag:
            dist = self.fy*obj_height/box_height/1000
            # self.estimate_distance_from_function(box_height)
        else:
            dist = 0
        
        return dist

    def estimate_distance_from_function(self, box_height):
        
        dist_function = ((box_height - 149.46)/-5.6858) * 25.4
        print("Ryan's Function distance: " + str(dist_function) + "mm")

        return None

    def estimate_obj_pos_in_world(self, dist, ucen, vcen, pose_w2b_W):
        
        (x_hat_C, y_hat_C, z_hat_C) = self.project_pixel_to_ray(ucen,vcen) 
        
        R_B2W = np.array([[ np.cos(pose_w2b_W[2]),  -np.sin(pose_w2b_W[2])],
                          [np.sin(pose_w2b_W[2]),  np.cos(pose_w2b_W[2])]]).reshape((2,2))
        R_W2B = R_B2W.T
        R_B2C = np.array([[ np.cos(self.base_to_camera[2]),  np.sin(self.base_to_camera[2])],
                          [-np.sin(self.base_to_camera[2]),  np.cos(self.base_to_camera[2])]]).reshape((2,2))
        R_C2B = R_B2C.T

        pos_w2b_W = pose_w2b_W[:2].reshape((2, 1)) 

        pos_b2c_B = np.array([self.base_to_camera[0], self.base_to_camera[1]]).reshape((2, 1))
        pos_b2c_C = R_B2C.dot(pos_b2c_B).reshape((2, 1))
        pos_c2o_C = dist*np.array([x_hat_C, z_hat_C]).reshape((2, 1))
        pos_b2o_C = pos_b2c_C.reshape((2,1)) + pos_c2o_C.reshape((2,1))

        pos_W = R_B2W.dot(R_C2B).dot(pos_b2o_C) + pos_w2b_W

        delta_theta = np.arctan2(x_hat_C, z_hat_C)
        # Goal theta = Current theta + delta theta
        theta_g = pose_w2b_W[2] - delta_theta

        return pos_W.reshape(2,), theta_g

    def camera_callback(self, msg):
        """ callback for camera images """

        # save the corresponding laser scan
        img_laser_ranges = list(self.laser_ranges)

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.camera_common(img_laser_ranges, img, img_bgr8)

    def compressed_camera_callback(self, msg):
        """ callback for camera images """

        # save the corresponding laser scan
        img_laser_ranges = list(self.laser_ranges)

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.camera_common(img_laser_ranges, img, img_bgr8)

    def camera_common(self, img_laser_ranges, img, img_bgr8):
        
        try:
            (translation,rotation) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            x_w2b_W = translation[0]
            y_w2b_W = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            theta = euler[2]
            pose_w2b_W = np.vstack((x_w2b_W, y_w2b_W, theta))
            print("Bacons world nav: " + str(pose_w2b_W[:2].flatten()))
            nav_flag = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            nav_flag = False

        (img_h,img_w,img_c) = img.shape

        # runs object detection in the image
        (boxes, scores, classes, num) = self.run_detection(img)

        if num > 0:
            # some objects were detected
            for (box,sc,cl) in zip(boxes, scores, classes):
                ymin = int(box[0]*img_h)
                xmin = int(box[1]*img_w)
                ymax = int(box[2]*img_h)
                xmax = int(box[3]*img_w)
                xcen = int(0.5*(xmax-xmin)+xmin)
                ycen = int(0.5*(ymax-ymin)+ymin)

                cv2.rectangle(img_bgr8, (xmin,ymin), (xmax,ymax), (255,0,0), 2)

                # computes the vectors in camera frame corresponding to each sides of the box
                rayleft = self.project_pixel_to_ray(xmin,ycen)
                rayright = self.project_pixel_to_ray(xmax,ycen)

                # convert the rays to angles (with 0 poiting forward for the robot)
                thetaleft = math.atan2(-rayleft[0],rayleft[2])
                thetaright = math.atan2(-rayright[0],rayright[2])
                if thetaleft<0:
                    thetaleft += 2.*math.pi
                if thetaright<0:
                    thetaright += 2.*math.pi

                box_height = np.abs(ymax - ymin)
                animal_labels = range(16, 26)
                animal_heights = [113,  # bird
                                  113,  # cat 
                                  113,  # dog
                                  113,  # horse
                                  113,  # sheep
                                  113,  # cow
                                  113,  # elephant
                                  113,  # bear
                                  113,  # zebra
                                  113   # giraffe
                                ]
                if cl == 13:
                    # Detected a stop sign
                    obj_height = 64 #mm
                    est_dist_flag = True
                elif cl in animal_labels:
                    # Detected an animal
                    obj_height = animal_heights[cl-16]
                    est_dist_flag = True
                else:
                    # Detected something else
                    obj_height = None
                    est_dist_flag = False

                # if cl == 'stop sign':
                dist = self.estimate_distance_from_image(box_height, obj_height, est_dist_flag)
                if nav_flag:
                    pos_obj_W, theta_g = self.estimate_obj_pos_in_world(dist, xcen, ycen, pose_w2b_W)
                    pos_obj_W_wflag = np.vstack((pos_obj_W.reshape(2,1), 1.0, theta_g)).flatten()
                else:
                    pos_obj_W = np.array([0, 0])
                    pos_obj_W_wflag = np.vstack((pos_obj_W.reshape((2,1)), 0.0, 0.0)).flatten()
                print("Object world pos: " + str(pos_obj_W))

                if not self.object_publishers.has_key(cl):
                    self.object_publishers[cl] = rospy.Publisher('/detector/'+self.object_labels[cl],
                        DetectedObject, queue_size=10)

                # publishes the detected object and its location
                object_msg = DetectedObject()
                object_msg.id = cl
                object_msg.name = self.object_labels[cl]
                object_msg.confidence = sc
                object_msg.distance = dist
                object_msg.thetaleft = thetaleft
                object_msg.thetaright = thetaright
                object_msg.corners = [ymin,xmin,ymax,xmax]
                object_msg.location_W = pos_obj_W_wflag.tolist()
                self.object_publishers[cl].publish(object_msg)

        # displays the camera image
        cv2.imshow("Camera", img_bgr8)
        cv2.waitKey(1)

    def camera_info_callback(self, msg):
        """ extracts relevant camera intrinsic parameters from the camera_info message.
        cx, cy are the center of the image in pixel (the principal point), fx and fy are
        the focal lengths. Stores the result in the class itself as self.cx, self.cy,
        self.fx and self.fy """

        K = msg.K
        self.cx = K[2]
        self.cy = K[5]
        self.fx = K[0]
        self.fy = K[4]

    def laser_callback(self, msg):
        """ callback for thr laser rangefinder """

        self.laser_ranges = msg.ranges
        self.laser_angle_increment = msg.angle_increment

    def run(self):
        rospy.spin()

if __name__=='__main__':
    d = Detector()
    d.run()
