import cv2
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
import collections.abc as collections

from functools import partial
from std_msgs.msg import Header
from gluestick_detection.msg import CustomFloatArray

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pytlsd import lsd
from pynput import keyboard
import time

callback_triggered = False
global image_width, image_height

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _numpy_to_multiarray(multiarray_type, np_array):
    multiarray = multiarray_type()
    multiarray.array.layout.dim = [MultiArrayDimension('dim%d' % i,
                                                 np_array.shape[i],
                                                 np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
    multiarray.array.data = np_array.reshape([1, -1])[0].tolist()
    return multiarray
# Get the current directory

# Change to your
# GLUESTICK_ROOT = "/home/jaeho/GlueStick/"
GLUESTICK_ROOT = rospy.get_param('gluestick_root')

if __name__ == '__main__':
	if __package__ is None:
		import sys
		from os import path
		# print(path.dirname( path.dirname( path.abspath(__file__) ) ))
		sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
		from models.two_view_pipeline import TwoViewPipeline
	else:
		from .two_view_pipeline import TwoViewPipeline
# Add it to the Python path
conf = {
    'name': 'two_view_pipeline',
    'use_lines': True,
    'extractor': {
        'name': 'wireframe',
        'sp_params': {
            'force_num_keypoints': False,
            'max_num_keypoints': 1000,
            'gluestick_root': GLUESTICK_ROOT
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
        },
        'max_n_lines': 300,
    },
    'matcher': {
        'name': 'gluestick',
        'weights': GLUESTICK_ROOT + "resources/weights/checkpoint_GlueStick_MD.tar",
        'trainable': False,
    },
    'ground_truth': {
        'from_pose_depth': False,
    }
}

# from two_view_pipeline import TwoViewPipeline

ref_line_pub = rospy.Publisher('/cam_1/lines', CustomFloatArray, queue_size=10, latch=True)
target_line_pub = rospy.Publisher('/cam_2/lines', CustomFloatArray, queue_size=10, latch=True)
ref_point_pub = rospy.Publisher('/cam_1/points', CustomFloatArray, queue_size=10, latch=True)
target_point_pub = rospy.Publisher('/cam_2/points', CustomFloatArray, queue_size=10, latch=True)
ref_rgb_pub = rospy.Publisher('/cam_1/rgb', Image, queue_size=10, latch=True)
ref_depth_pub = rospy.Publisher('cam_1/depth', Image, queue_size=10, latch=True)
target_rgb_pub = rospy.Publisher('cam_2/rgb', Image, queue_size=10, latch=True)
target_depth_pub = rospy.Publisher('cam_2/depth', Image, queue_size=10, latch=True)

def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()


def map_tensor(input_, func):
    if isinstance(input_, (str, bytes)):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        return func(input_)


def batch_to_np(batch):
    return map_tensor(batch, lambda t: t.detach().cpu().numpy()[0])

pipeline_model = TwoViewPipeline(conf).to(device).eval()
import gc

def line_detection(gray0, gray1):
    global image_width

    torch.cuda.empty_cache()
    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}
    with torch.no_grad():
        pred = pipeline_model(x)
    pred = batch_to_np(pred)
    # Matching lines

    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
    line_matches = pred["line_matches0"]    
    valid_matches = line_matches != -1
    # valid_matches = pred["line_match_scores0"] > 0.6
    match_indices = line_matches[valid_matches]
    matched_lines0 = line_seg0[valid_matches]
    matched_lines1 = line_seg1[match_indices]
    matched_num = matched_lines0.shape[0]

    # print(matched_lines0)
    ref_matches_list = []
    target_matches_list = []
    for i in range(matched_num):
        # is_long = np.linalg.norm(matched_lines0[i][0] - matched_lines0[i][1]) > 100 and  np.linalg.norm(matched_lines1[i][0] - matched_lines1[i][1]) > 100
        is_long = np.linalg.norm(matched_lines0[i][0] - matched_lines0[i][1]) > 300 or  np.linalg.norm(matched_lines1[i][0] - matched_lines1[i][1]) > 300

        sp1_continuous = np.abs(matched_lines0[i][0][0]) > 0 and np.abs(matched_lines0[i][0][0]) < 30
        ep1_continuous = np.abs(matched_lines0[i][1][0]) > 0 and np.abs(matched_lines0[i][1][0]) < 30
        sp2_continuous = np.abs(matched_lines1[i][0][0]) > (image_width- 10) and np.abs(matched_lines1[i][0][0]) < image_width
        ep2_continuous = np.abs(matched_lines1[i][1][0]) > (image_width- 10) and np.abs(matched_lines1[i][1][0]) < image_width
            
        if(is_long and (sp1_continuous or ep1_continuous) and (sp2_continuous or ep2_continuous)):
            ref_line = np.array([[matched_lines0[i][0][0], matched_lines0[i][0][1]], [matched_lines0[i][1][0], matched_lines0[i][1][1]]]) 
            target_line = np.array([[matched_lines1[i][0][0], matched_lines1[i][0][1]], [matched_lines1[i][1][0], matched_lines1[i][1][1]]])
            ref_matches_list.append(ref_line)
            target_matches_list.append(target_line)


    del torch_gray0, torch_gray1, x, pred
    torch.cuda.empty_cache()
    gc.collect()
    return np.array(ref_matches_list,dtype=np.float32), np.array(target_matches_list,dtype=np.float32)


# Publish only depth
def publish_message(ref_line,target_line,ref_depth,target_depth):
    
    n = ref_line.shape[0]
    if(n>0):
        ref_line_msg = CustomFloatArray()
        ref_line_msg.header = Header(stamp=rospy.Time.now())

        ref_line_msg.data = ref_line.flatten().tolist()  # Convert the 3D array to a 1D list
        ref_line_msg.dim1 = n
        ref_line_msg.dim2 = 2
        ref_line_msg.dim3 = 2

        target_line_msg = CustomFloatArray()
        target_line_msg.header = Header(stamp=rospy.Time.now())

        target_line_msg.data = target_line.flatten().tolist()  # Convert the 3D array to a 1D list
        target_line_msg.dim1 = n
        target_line_msg.dim2 = 2
        target_line_msg.dim3 = 2



        ref_depth_msg = Image()
        ref_depth_msg.header = Header(stamp=rospy.Time.now())
        ref_depth_msg.height = ref_depth.shape[0]
        ref_depth_msg.width = ref_depth.shape[1]
        ref_depth_msg.encoding = "16UC1"
        ref_depth_msg.is_bigendian = 0  # Assumes the system's byte order is little-endian
        ref_depth_msg.step = ref_depth_msg.width * 1 * 2  # Full row length in bytes
        ref_depth_msg.data = ref_depth.tobytes()

    

        target_depth_msg = Image()
        target_depth_msg.header = Header(stamp=rospy.Time.now())
        target_depth_msg.height = target_depth.shape[0]
        target_depth_msg.width = target_depth.shape[1]
        target_depth_msg.encoding = "16UC1"
        target_depth_msg.is_bigendian = 0  # Assumes the system's byte order is little-endian
        target_depth_msg.step = target_depth_msg.width * 1 * 2  # Full row length in bytes
        target_depth_msg.data = target_depth.tobytes()


        # Publish void point message
        ref_points_msg = CustomFloatArray()
        ref_points_msg.header = Header(stamp=rospy.Time.now())

        ref_points_msg.data = [] # Convert the 3D array to a 1D list
        ref_points_msg.dim1 = 0
        ref_points_msg.dim2 = 0
        ref_points_msg.dim3 = 0

        target_points_msg = CustomFloatArray()
        target_points_msg.header = Header(stamp=rospy.Time.now())

        target_points_msg.data = []  # Convert the 3D array to a 1D list
        target_points_msg.dim1 = 0
        target_points_msg.dim2 = 0
        target_points_msg.dim3 = 0
        # Publish Message
        ref_line_pub.publish(ref_line_msg)
        target_line_pub.publish(target_line_msg)
        ref_depth_pub.publish(ref_depth_msg)
        target_depth_pub.publish(target_depth_msg)
        ref_point_pub.publish(ref_points_msg)
        target_point_pub.publish(target_points_msg)


bridge = CvBridge()


def callback(rgb_ref_msg, depth_ref_msg, rgb_target_msg, depth_target_msg):
    global callback_triggered

    if(callback_triggered == True):

        rgb_ref = bridge.imgmsg_to_cv2(rgb_ref_msg, "bgr8")
        depth_ref = bridge.imgmsg_to_cv2(depth_ref_msg, desired_encoding="passthrough")
        rgb_target = bridge.imgmsg_to_cv2(rgb_target_msg, "bgr8")
        depth_target = bridge.imgmsg_to_cv2(depth_target_msg, desired_encoding="passthrough")

        gray_ref = cv2.cvtColor(rgb_ref, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(rgb_target, cv2.COLOR_BGR2GRAY)


        ref_lines, target_lines = line_detection(gray_ref, gray_target)
        publish_message(ref_lines, target_lines, depth_ref, depth_target)
        callback_triggered = False



def main():
    global callback_triggered, image_width, image_height

    rospy.init_node('line_img_subscriber', anonymous=True)

    image_width = rospy.get_param('image_width')
    image_height = rospy.get_param('image_height')
    target_rgb_topic = rospy.get_param('target_rgb')
    target_depth_topic = rospy.get_param('target_depth')
    source_rgb_topic = rospy.get_param('source_rgb')
    source_depth_topic = rospy.get_param('source_depth')

    rgb_ref_sub = message_filters.Subscriber(source_rgb_topic, Image)
    depth_ref_sub = message_filters.Subscriber(source_depth_topic, Image)
    rgb_target_sub = message_filters.Subscriber(target_rgb_topic, Image)
    depth_target_sub = message_filters.Subscriber(target_depth_topic, Image)


    ats = message_filters.ApproximateTimeSynchronizer([rgb_ref_sub, depth_ref_sub, rgb_target_sub, depth_target_sub], 100, 0.1)
    ats.registerCallback(callback)    

    while not rospy.is_shutdown():
        callback_triggered = True
        rospy.sleep(0.8)

if __name__ == "__main__":
    main()

