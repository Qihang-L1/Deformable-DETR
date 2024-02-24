# The original code is provided by Capgemini Engineering Germany
# to all partners within the KI-WISSEN project.
# The code may be used and modified without restriction and on own risk

import rospy
import json
import numpy as np
import cv2
import skimage.io
import os
import signal
import sys

import threading
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from carla_msgs.msg import CarlaControl
from tf2_msgs.msg import TFMessage
from derived_object_msgs.msg import Object
from derived_object_msgs.msg import ObjectArray

import time
import struct
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse

import carla  # the CARLA API in ONLY needed for visual debugging!!! (check world, client and world.debug and remove if not needed)
from multipleobjecttracking import tracker, kalmanFilter

# declare global variables
# bbox_sizes is rough approximation of the bounding boxes for
# 1: pedestrian
# 2: vehicle
# The AD-Stack does not need the true bbox of an actor (ex. vehicle)
# instead it only needs the relevant OBSTACLE bbox, where obstacle can be just (and will be for 2d->3d) a portion of detected actor
bbox_sizes = {1: [0.5, 0.5, 2.0], 2: [2.0, 2.0, 2.0]}
# the carla API is only needed for debugging
world = None
doLoop = True
rgb_message = None
prio_message = None
depth_message = None
dc_info = None
# Attention! tf message is in global CS! See comment to MapToCamProjector.cc (function LocalizationCallback)
tf_message = None
condition_object = threading.Condition()
timestamp = 0
bboxes = []
masks = []
trk = tracker.Tracker(5, 30)

# the function will prepare the message of type Object
# (http://docs.ros.org/en/melodic/api/derived_object_msgs/html/msg/Object.html)
# and will collect the 3d points for visualization in Carla


def make_ros_bbox(bbox2d, center, cam_trf):
    global bbox_sizes
    v = center - cam_trf[:, 3]  # we lost here the homomorphic 1
    b = bbox_sizes[bbox2d[0]]  # estimate_3dbbox(bbox2d, v, cam_trf)
    dir = v / np.linalg.norm(v)
    center = np.add(center, 0.5 * dir * b[0])

    viz_points = []
    M = np.eye(4)
    M[0:3, 3] = [center[0], -center[1], center[2]]
    M[0:3, 0:3] = R.from_euler("xyz", [0, 0, 0], degrees=False).as_matrix()
    i = -0.5 * b[0]
    while i < 0.5 * b[0]:
        viz_points.append(M.dot([+i, -0.5 * b[1], -0.5 * b[2], 1]))
        viz_points.append(M.dot([+i, +0.5 * b[1], -0.5 * b[2], 1]))
        viz_points.append(M.dot([+i, -0.5 * b[1], +0.5 * b[2], 1]))
        viz_points.append(M.dot([+i, +0.5 * b[1], +0.5 * b[2], 1]))
        i += 0.1
    i = -0.5 * b[1]
    while i < 0.5 * b[1]:
        viz_points.append(M.dot([-0.5 * b[0], +i, -0.5 * b[2], 1]))
        viz_points.append(M.dot([+0.5 * b[0], +i, -0.5 * b[2], 1]))
        viz_points.append(M.dot([-0.5 * b[0], +i, +0.5 * b[2], 1]))
        viz_points.append(M.dot([+0.5 * b[0], +i, +0.5 * b[2], 1]))
        i += 0.1
    i = -0.5 * b[2]
    while i < 0.5 * b[2]:
        viz_points.append(M.dot([-0.5 * b[0], -0.5 * b[1], +i, 1]))
        viz_points.append(M.dot([+0.5 * b[0], -0.5 * b[1], +i, 1]))
        viz_points.append(M.dot([-0.5 * b[0], +0.5 * b[1], +i, 1]))
        viz_points.append(M.dot([+0.5 * b[0], +0.5 * b[1], +i, 1]))
        i += 0.1

    object = Object()
    object.detection_level = 0  # OBJECT_DETECTED
    object.classification = bbox2d[0]
    object.pose.position.x = center[0]
    object.pose.position.y = center[1]
    object.pose.position.z = center[2]
    if bbox2d[0] == 1:
        object.classification = 4
    if bbox2d[0] == 2:
        object.classification = 6
    object.shape.dimensions.append(b[0])
    object.shape.dimensions.append(b[1])
    object.shape.dimensions.append(b[2])
    # we skip orientation now
    return viz_points, object


# the function will return the center of 3d object corresponding to the 2d object (bb,m)
# camTrf is camera transformation in World


def getCenter(bb, m, camTrf):
    # we will analyse mask for better precision! (not bbox!)
    global depth_message, dc_info, bbox_sizes
    # eventhough we are using mask and not bbox to analyse pixels
    # there will be outliers since not all pixels in the mask actually belong to object!
    # also the possibility exists that there will be another object in the foreground (ex. lamppost in front of detected car)
    # we will try to filter out these outliers by applying median filter:
    points = []  # first we collect all points as tuple (key=depth, value=[x,y,z])
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if m[r, c]:  # if 1(True)
                R = r + bb[5]
                C = c + bb[4]
                d = struct.unpack(
                    "f",
                    depth_message.data[
                        4 * (R * dc_info.width + C) : 4 * (R * dc_info.width + C) + 4
                    ],
                )[0]
                v = [
                    float(C - dc_info.K[2]) / dc_info.K[0] * d,
                    float(R - dc_info.K[5]) / dc_info.K[4] * d,
                    d,
                    1,
                ]  # 3d points in camera space(Lqh)
                v = camTrf.dot(v)  # transform into world cs
                points.append((d, v))

    # now we sort the points by the depth:
    points.sort(key=lambda a: a[0])
    # take the medium (middle point):
    index = len(points) // 2
    middle = points[index]
    # retain only the points that are within the the bbox size
    medians = []
    while index < len(points):  # from camera
        p = points[index]
        if abs(p[0] - middle[0]) > bbox_sizes[bb[0]][0]:
            break
        medians.append(p[1])
        index += 1
    index = len(points) // 2 - 1
    while index > 0:  # to camera
        p = points[index]
        if abs(p[0] - middle[0]) > bbox_sizes[bb[0]][0]:
            break
        medians.append(p[1])
        index -= 1

    center = [0, 0, 0, 1]

    # find the center, and collect X,Y,Z for ground filtering
    X = []
    Y = []
    Z = []
    D = []
    for v in medians:
        X.append(v[0])
        Y.append(v[1])
        Z.append(v[2])
        center = [center[0] + v[0], center[1] + v[1], center[2] + v[2], 1]
    a = np.array([X, Y, Z])
    c = np.cov(a)
    try:
        eigen_values, eigen_vectors = np.linalg.eig(c)
    except:
        pass  # ignore

    center = [
        center[0] / len(medians),
        center[1] / len(medians),
        center[2] / len(medians),
        1,
    ]
    # if the smallest component is pointing alone Z and the PC is flat:

    # We agreed that we will not use additional filtering of 3d objects
    # based on the augmented (depth) information
    # i.e. we should propagate the 2D noise into 3D as is and not suppress it
    if False:
        if abs(eigen_vectors[-1][2]) > 0.99 and eigen_values[-1] < 0.01:
            return center, False  # flat horizontal ground
    # visualize in Carla simulator the median 3d points that correspond to the object:
    if False:
        for x, y, z in zip(X, Y, Z):
            # visual debugging may be very slow, since may be rasterizing large arears and may time-out ros-bridge
            world.debug.draw_string(
                carla.Location(x, -y, z), "*", False, carla.Color(0, 0, 255)
            )

    return center, True


def track_objects(objects):
    global timestamp, trk
    centers = dict()
    for i in range(len(objects.objects)):
        centers[i] = np.array(
            [objects.objects[i].pose.position.x, objects.objects[i].pose.position.y]
        )
    trk.update(float(timestamp) * 0.000000001, centers)
    for track in trk.tracks:
        obj_id = track.trace[-1][0]
        if obj_id != -1:
            objects.objects[obj_id].id = track.trackId
            objects.objects[obj_id].detection_level = 1  # OBJECT_TRACKED

    if True:  # visualize tracking
        for track in trk.tracks:
            rgb = [
                track.trackId * 71 % 256,
                track.trackId * 103 % 256,
                track.trackId * 137 % 256,
            ]
            for val in track.trace:
                if val[0] != -1:
                    pos = [val[1][0, 0], val[1][0, 1], 1]
                    # eventually a bug in world.debug, since only monochrome colors can be rendered
                    world.debug.draw_string(
                        carla.Location(pos[0], -pos[1], pos[2]),
                        "*",
                        False,
                        carla.Color(rgb[0], rgb[1], rgb[2]),
                    )
            world.debug.draw_string(
                carla.Location(track.trace[-1][1][0, 0], -track.trace[-1][1][0, 1], 1),
                "ID=" + str(track.trackId),
                False,
                carla.Color(0, 0, 0),
            )


# the function converts the 2d bboxes into 3d bboxes
# creates array of ros Objects (ObjectArray) and
# tracks the objects:
def convert2dto3d():
    global doLoop, bboxes, depth_message, rgb_message, dc_info, tf_message, world, masks
    publisher = rospy.Publisher("testing_bboxes", Image, queue_size=10)

    # visualization of 2d bboxes:
    # these 2D bounding box
    if True:
        rgba_image = np.array(list(rgb_message.data)).reshape((720, 1280, 4))
        for bbox in bboxes:
            rgba_image[bbox[5] - 2 : bbox[5] + 2, bbox[4] : bbox[6], 0] = 0
            rgba_image[bbox[5] - 2 : bbox[5] + 2, bbox[4] : bbox[6], 1] = 0
            rgba_image[bbox[5] - 2 : bbox[5] + 2, bbox[4] : bbox[6], 2] = 255
            rgba_image[bbox[7] - 2 : bbox[7] + 2, bbox[4] : bbox[6], 0] = 0
            rgba_image[bbox[7] - 2 : bbox[7] + 2, bbox[4] : bbox[6], 1] = 0
            rgba_image[bbox[7] - 2 : bbox[7] + 2, bbox[4] : bbox[6], 2] = 255

            rgba_image[bbox[5] : bbox[7], bbox[4] - 2 : bbox[4] + 2, 0] = 0
            rgba_image[bbox[5] : bbox[7], bbox[4] - 2 : bbox[4] + 2, 1] = 0
            rgba_image[bbox[5] : bbox[7], bbox[4] - 2 : bbox[4] + 2, 2] = 255
            rgba_image[bbox[5] : bbox[7], bbox[6] - 2 : bbox[6] + 2, 0] = 0
            rgba_image[bbox[5] : bbox[7], bbox[6] - 2 : bbox[6] + 2, 1] = 0
            rgba_image[bbox[5] : bbox[7], bbox[6] - 2 : bbox[6] + 2, 2] = 255

        # visualization of 2d bboxes centers:
        if True:
            for bbox in cbboxes:
                c = int(0.5 * (bbox[4] + bbox[6]))
                r = int(0.5 * (bbox[5] + bbox[7]))
                rgba_image[r - 2 : r + 2, c - 10 : c + 10, 0] = 255
                rgba_image[r - 2 : r + 2, c - 10 : c + 10, 1] = 0
                rgba_image[r - 2 : r + 2, c - 10 : c + 10, 2] = 0
                rgba_image[r - 10 : r + 10, c - 2 : c + 2, 0] = 255
                rgba_image[r - 10 : r + 10, c - 2 : c + 2, 1] = 0
                rgba_image[r - 10 : r + 10, c - 2 : c + 2, 2] = 0

        rgb_message.data = bytearray(rgba_image.reshape(-1).astype(int).tolist())
        publisher.publish(rgb_message)

    # we also could use the Carla API (however will probably not, since we must only subscribe to carla-ros messages):
    if True:
        actors = world.get_actors()
        ego = None
        for actor in actors:
            if actor.attributes["role_name"] == "hero":
                ego = actor
                break

        if ego is None:
            return
        # in case the direct Carla API is used, we can aquire the ego transformation
        trl = ego.get_transform().location
        trf = np.eye(4)
        trf[0:3, 3] = [trl.x, trl.y, trl.z]
        rot = ego.get_transform().rotation
        trf[0:3, 0:3] = R.from_euler(
            "zyx", [rot.yaw, rot.pitch, rot.roll], degrees=True
        ).as_matrix()

    # depth camera extrinsic transformation entry inside the tf message
    # ATTENTION: the carla-ros-bridge is providing the tf of attached camera in global CS!!!
    cam_tf_ros = None
    for tf in tf_message.transforms:
        if tf.child_frame_id == "hero/camera/depth/CAM_0":
            cam_tf_ros = tf
            break

    if cam_tf_ros is None:
        print("Camera transformation not found ************")
        return

    # get camera transformation in World:
    trl = cam_tf_ros.transform.translation
    cam_tf = np.eye(4)
    cam_tf[0:3, 3] = [trl.x, trl.y, trl.z]
    q = cam_tf_ros.transform.rotation
    cam_tf[0:3, 0:3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    # create the ros ObjectArray message
    objects = ObjectArray()

    # for each 2d detection get center of the corresponding 3d object
    for bb, m in zip(bboxes, masks):
        center, isNotGround = getCenter(bb, m, cam_tf)
        if not isNotGround:
            continue

        # world.debug.draw_box(carla.BoundingBox(carla.Location(v[0],v[1],v[2]),carla.Vector3D(1,1,1)),carla.Rotation(0,0,0)) # buggy, memory leak
        points, o = make_ros_bbox(bb, center, cam_tf)
        objects.objects.append(o)
        if True:  # visualize bounding boxes in Carla:
            for p in points:
                world.debug.draw_string(
                    carla.Location(p[0], p[1], p[2]), "*", False, carla.Color(255, 0, 0)
                )

    track_objects(objects)

    return objects


def inference():
    publisher3 = rospy.Publisher("inference/hero/objects", ObjectArray, queue_size=10)
    time.sleep(2)
    carla_control_msg = CarlaControl()
    carla_control_msg.command = 2
    global rgb_message, prio_message, depth_message, tf_message, parser, masks, bboxes
    global world

    print("inference")

    while doLoop:
        condition_object.acquire()
        if rgb_message is None or prio_message is None:
            condition_object.wait()
        if not doLoop:
            break

        if world is None:
            # the carla API is only needed for debugging
            # these two lines are for visualizing 3D bounding boxs of objects in real time, which would require carla API  the final version of the project should not include these lines ps:world.helper.debug() (Lqh)
            client = carla.Client("127.0.0.1", 2000)
            world = client.get_world()

        # DO HERE anything that needs to be done in sequire section
        condition_object.release()

        bboxes, masks = AI_2D_ObjectDetection()  # 2D object detection module
        # here comes the main dish! video 0:08:50(Lqh)
        # len(bboxes) == len(masks)
        # bboxes is a list. Each entry of bboxes is in turn a list with values:
        # [class_id, instance_id, score, x1, y1, x2, y2]
        # they use the coordinates of left-most and right-most corners of the bounding box(Lqh)

        # masks is a list. Each entry of masks is a matrix of size of bbox (x1,y1,x2,y2) with values:
        # 1 (True) correspond to object
        # 0 (False) corresponding to no object
        # according to the video 0:10:00, the mask is referred to a more precise specification of the objects, not in cuboid form. However,flase positive is always a big problem.(Lqh)

        # one mask example:

        #   x1            x2
        #   0 0 0 1 0 0 0 0  y2
        #   0 1 1 1 1 1 0 0
        #   0 1 1 1 1 1 1 0
        #   0 0 1 1 1 1 0 0
        #   0 0 1 1 1 0 0 0
        #   0 0 0 1 1 0 0 0  y1

        if not doLoop:
            break

        objects = convert2dto3d()
        publisher3.publish(objects)

        if not doLoop:
            break

        condition_object.acquire()
        rgb_message = None
        prio_message = None
        # in case we want to close the loop in our code (without sending any signal to AVL AD stack)
        # we need to publish the control message now:
        # print("publishing control message")
        # publisher2.publish(carla_control_msg)
        condition_object.release()


def rgb_callback(data):
    print("Received RGB")
    global rgb_message, prio_message, timestamp
    condition_object.acquire()
    timestamp = data.header.stamp.secs * 1000000000 + data.header.stamp.nsecs
    rgb_message = data
    # print(data.height)
    # print(type(data.width))
    # print(dir(data.encoding))

    # # rosmsg_cv2
    # cv_image = np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 4))
    # cv2.imwrite("/home/carla/carla_rgb_callback.png", cv_image[:,:,:3])
    # print(type(cv_image))
    # print(cv_image.shape)

    if rgb_message is not None and prio_message is not None:
        condition_object.notify()
    condition_object.release()


def prior_callback(data):
    # The prior image is similar to segmentation image with some additional information(Lqh)
    print("Received Prio")
    global rgb_message, prio_message
    condition_object.acquire()
    prio_message = data
    if rgb_message is not None and prio_message is not None:
        condition_object.notify()
    condition_object.release()


def depth_callback(data):
    print("Received Depth")
    global depth_message
    depth_message = data


def depth_camera_info_callback(data):
    # The depth camera and the RGB camera are of the same position and same orientation, extrinsic callibration is not taken between these sensors.(Lqh)
    global dc_info
    if dc_info is None:
        # lets keep it simple, camera info will not change over time
        dc_info = data


def tf_callback(data):
    print("Received tf")
    global tf_message
    tf_message = data


def signal_handler(sig, frame):
    print("Gracefully quitting...")
    global doLoop
    doLoop = True
    condition_object.acquire()
    condition_object.notify()
    condition_object.release()


def AI_2D_ObjectDetection():
    data = rgb_message
    image_count = 0
    cv_image = np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 4))
    # cv2.imwrite("/home/carla/carla_rgb_message.png", cv_image[:,:,:3])
    if image_count < 50:
        image_filename = "/home/carla/Documents/Dataset/image{}.png".format(image_count)
        cv2.imwrite(image_filename, cv_image[:, :, :3])
        image_count += 1

    """
    cv_image = imgmsg_to_cv2(rgb_message,desired_encoding = 'passthrough')
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    height,width,channels = img.shape
    rospy.loginfo(height/n)
    rospy.loginfo(width/n)
    rospy.loginfo(channels/n)
    bbox = [[4,1,0.9,100,100,200,200],[6,1,0.8,150,150,200,200]] # [class_id, instance_id, score, x1, y1, x2, y2]
    mask = [np.ones(100,100),np.ones(50,50)] # masks is a list. Each entry of masks is a matrix of size of bbox (x1,y1,x2,y2) with values:
        # 1 (True) correspond to object
        # 0 (False) corresponding to no object
    return bbox, mask
    """
    pass


def imgmsg_to_cv2(data, desired_encoding="passthrough"):
    """
    Converts a ROS image to an OpenCV image without using the cv_bridge package,
    for compatibility purposes.
    """

    flip_channels = False

    if desired_encoding == "passthrough":
        encoding = data.encoding
    else:
        encoding = desired_encoding

    if encoding == "bgr8" or (encoding == "rgb8" and flip_channels):
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 3))
    elif encoding == "rgb8" or (encoding == "bgr8" and flip_channels):
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 3))[
            :, :, ::-1
        ]
    elif encoding == "mono8" or encoding == "8UC1":
        return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width))
    elif encoding == "mono16" or encoding == "16UC1":
        return np.frombuffer(data.data, np.uint16).reshape((data.height, data.width))
    else:
        rospy.logwarn("Unsupported encoding %s" % encoding)
        return None


# def thread_spin():
#   rospy.spin()
def start():
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("listener", anonymous=True)
    rospy.Subscriber("/carla/hero/camera/rgb/CAM_0/image_color", Image, rgb_callback)
    rospy.Subscriber("/CAM_0/prior/image_raw", Image, prior_callback)
    rospy.Subscriber(
        "/carla/hero/camera/depth/CAM_0/image_depth", Image, depth_callback
    )
    rospy.Subscriber(
        "/carla/hero/camera/depth/CAM_0/camera_info",
        CameraInfo,
        depth_camera_info_callback,
    )
    rospy.Subscriber("/tf", TFMessage, tf_callback)
    t1 = threading.Thread(target=inference)
    # t2 = threading.Thread(target = thread_spin)
    t1.start()
    # t2.start()
    t1.join()
    # t2.join()
    rospy.spinOnce()


start()
