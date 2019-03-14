# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import cv2
import numpy as np

from joint import Joint
jta_idx = [i for i in range(22)]
#created with:

crowd_pose_idx = Joint.get_conversion_idx(Joint.NAMES, Joint.NAMES_CROWDPOSE)#[8, 4, 9, 5, 10, 6, 19, 16, 20, 17, 21, 18, 0, 2]
pose_track_idx = Joint.get_conversion_idx(Joint.NAMES, Joint.NAMES_POSETRACK)
class Pose(list):
	"""
	a Pose is a list of Joint(s) belonging to the same person.
	"""



	LIMBS = [
		(0, 1),  # head_top -> head_center
		(1, 2),  # head_center -> neck
		(2, 3),  # neck -> right_clavicle
		(3, 4),  # right_clavicle -> right_shoulder
		(4, 5),  # right_shoulder -> right_elbow
		(5, 6),  # right_elbow -> right_wrist
		(2, 7),  # neck -> left_clavicle
		(7, 8),  # left_clavicle -> left_shoulder
		(8, 9),  # left_shoulder -> left_elbow
		(9, 10),  # left_elbow -> left_wrist
		(2, 11),  # neck -> spine0
		(11, 12),  # spine0 -> spine1
		(12, 13),  # spine1 -> spine2
		(13, 14),  # spine2 -> spine3
		(14, 15),  # spine3 -> spine4
		(15, 16),  # spine4 -> right_hip
		(16, 17),  # right_hip -> right_knee
		(17, 18),  # right_knee -> right_ankle
		(15, 19),  # spine4 -> left_hip
		(19, 20),  # left_hip -> left_knee
		(20, 21)  # left_knee -> left_ankle
	]

	SKELETON = [[l[0] + 1, l[1] + 1] for l in LIMBS]
	#copied skeleton from crowdpose annotations but it seems nonsensical
	SKELETON_CROWD_POSE = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]

	def __init__(self, joints, keypoint_style):
		# type: (List[Joint]) -> None
		super().__init__(joints)
		self.keypoint_style = keypoint_style
		self.n = 5
		self.dist_thresh = 40

	@property
	def invisible(self):
		# type: () -> bool
		"""
		:return: True if all the joints of the pose are occluded, False otherwise
		"""
		for j in self:
			if not j.occ:
				return False
		return True

	@property
	def too_far_from_camera(self):

		for j in self:
			if j.cam_distance > self.dist_thresh:
				return True
		return False


	@property
	def visible_and_onscreen_atleast_n(self):
		# type: () -> bool
		"""
		:return: True if at least n joints are on creen and not occluded
		"""
		a = 0
		for j in self:
			if j.is_on_screen and not j.occ:
				a = a +1
			if a >= self.n:
				return True

		return False


	@property
	def bbox_2d(self):
		# type: () -> List[int]
		"""
		:return: bounding box around the pose in format [x_min, y_min, width, height]
			- x_min = x of the top left corner of the bounding box
			- y_min = y of the top left corner of the bounding box
		"""
		x_min = int(np.min([j.x2d for j in self]))
		y_min = int(np.min([j.y2d for j in self]))
		x_max = int(np.max([j.x2d for j in self]))
		y_max = int(np.max([j.y2d for j in self]))
		width = x_max - x_min
		height = y_max - y_min
		return [x_min, y_min, width, height]


	@property
	def coco_annotation(self):
		# type: () -> Dict
		"""
		:return: COCO annotation dictionary of the pose
		==========================================================
		NOTE#1: in COCO, each keypoint is represented by its (x,y)
		2D location and a visibility flag `v` defined as:
			- `v=0` ==> not labeled (in which case x=y=0)
			- `v=1` ==> labeled but not visible
			- `v=2` ==> labeled and visible
		==========================================================
		NOTE#2: in COCO, a keypoint is considered visible if it
		falls inside the object segment. In JTA there are no
		object segments and every keypoint is labelled, so we
		v=2 for each keypoint.
		==========================================================
		"""
		keypoints = []
		for j in self:
			keypoints += [j.x2d, j.y2d, 2]
		annotation = {
			'keypoints': keypoints,
			'num_keypoints': len(self),
			'bbox': self.bbox_2d
		}
		return annotation

	@property
	def dino_annotation(self):
		# type: () -> Dict
		"""
		:return: COCOlike annotation dictionary of the pose
		==========================================================
		NOTE#1: in ###, each keypoint is represented by its (x,y)
		2D location and a visibility flag `v` defined as:
			- `v=0` ==> not labeled (in which case x=y=0)
			- `v=1` ==> labeled but not visible
			- `v=2` ==> labeled and visible
			- 'v=3' ==> labeld and self occluded
		==========================================================
		"""
		if(self.keypoint_style == "CrowdPose"):
			idx = crowd_pose_idx
		elif(self.keypoint_style == "PoseTrack"):
			idx = pose_track_idx
		else:
			idx = jta_idx

		kpts = np.zeros((len(self), 3))
		for i, j in enumerate(self):
			if j.is_on_screen:
				kpts[i, ] = [j.x2d, j.y2d, j.get_v_flag]
		keypoints = np.zeros((len(idx), 3))
		for k, i in enumerate(idx):
			keypoints[k,] = kpts[i, ]

		annotation = {
			'keypoints': keypoints.tolist(),
			'num_keypoints': len(idx),
			'bbox': self.bbox_2d
		}
		return annotation


	def draw(self, image, color):
		# type: (np.ndarray, List[int]) -> np.ndarray
		"""
		:param image: image on which to draw the pose
		:param color: color of the limbs make up the pose
		:return: image with the pose
		"""
		# draw limb(s) segments
		for (j_id_a, j_id_b) in Pose.LIMBS:
			joint_a = self[j_id_a]  # type: Joint
			joint_b = self[j_id_b]  # type: Joint
			t = 1 if joint_a.cam_distance > 25 else 2
			if joint_a.is_on_screen and joint_b.is_on_screen:
				cv2.line(image, joint_a.pos2d, joint_b.pos2d, color=color, thickness=t)

		# draw joint(s) circles
		for joint in self:
			image = joint.draw(image)

		return image


	def __iter__(self):
		# type: () -> Iterator[Joint]
		return super().__iter__()
