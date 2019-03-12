# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import cv2
import numpy as np


class Joint(object):
	"""
	a Joint is a keypoint of the human body.
	"""

	# list of joint names
	NAMES = [
		'head_top',
		'head_center',
		'neck',
		'right_clavicle',
		'right_shoulder',
		'right_elbow',
		'right_wrist',
		'left_clavicle',
		'left_shoulder',
		'left_elbow',
		'left_wrist',
		'spine0',
		'spine1',
		'spine2',
		'spine3',
		'spine4',
		'right_hip',
		'right_knee',
		'right_ankle',
		'left_hip',
		'left_knee',
		'left_ankle',
	]
	#same as in annotations but head replaced with head_top
	NAMES_CROWDPOSE = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
					   'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
					   'left_ankle', 'right_ankle', 'head_top', 'neck']

	# same as in annotations but head_bottom replaced with neck
	NAMES_POSETRACK = ['nose', 'neck', 'head_top', 'left_ear', 'right_ear', 'left_shoulder',
					   'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
					   'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

	def __init__(self, array):
		# type: (np.ndarray) -> None
		"""
		:param array: array version of the joint
		"""
		self.frame = int(array[0])
		self.person_id = int(array[1])
		self.type = int(array[2])
		self.x2d = int(array[3])
		self.y2d = int(array[4])
		self.x3d = array[5]
		self.y3d = array[6]
		self.z3d = array[7]
		self.occ = bool(array[8])  # is this joint occluded?
		self.soc = bool(array[9])  # is this joint self-occluded?


	@property
	def cam_distance(self):
		# type: () -> float
		"""
		:return: distance of the joint from the camera
		"""
		# NOTE: camera coords = (0, 0, 0)
		return np.sqrt(self.x3d**2 + self.y3d**2 + self.z3d**2)


	@property
	def is_on_screen(self):
		# type: () -> bool
		"""
		:return: True if the joint is on screen, False otherwise
		"""
		return (0 <= self.x2d <= 1920) and (0 <= self.y2d <= 1080)


	@property
	def visible(self):
		# type: () -> bool
		"""
		:return: True if the joint is visible, False otherwise
		"""
		return not (self.occ or self.soc)

	@property
	def get_v_flag(self):
		# type: () -> int
		"""
		:return: Return Visbility Flag
		"""
		if (self.occ):
			return 1  # labeled but not visible
		if (self.soc):
			return 3  # labeled but occluded by self
		return 2  # labeled and visible

	@property
	def pos2d(self):
		# type: () -> Tuple[int, int]
		"""
		:return: 2D coordinates of the joints [px]
		"""
		return (self.x2d, self.y2d)


	@property
	def pos3d(self):
		# type: () -> Tuple[float, float, float]
		"""
		:return: 3D coordinates of the joints [m]
		"""
		return (self.x3d, self.y3d, self.z3d)


	@property
	def color(self):
		# type: () -> Tuple[int, int, int]
		"""
		:return: the color with which to draw the joint;
		this color is chosen based on the visibility of the joint:
		(1) occluded joint --> RED
		(2) self-occluded joint --> ORANGE
		(2) visible joint --> GREEN
		"""
		if self.occ:
			return (255, 0, 42)  # red
		elif self.soc:
			return (255, 128, 42)  # orange
		else:
			return (0, 255, 42)  # green


	@property
	def radius(self):
		# type: () -> int
		"""
		:return: appropriate radius [px] for the circle that represents the joint;
		this radius is a function of the distance of the joint from the camera
		"""
		radius = int(round(np.power(10, 1 - (self.cam_distance/20.0))))
		return radius if radius >= 1 else 1


	@property
	def name(self):
		# type: () -> str
		"""
		:return: name of the joint (eg: 'neck', 'left_elbow', ...)
		"""
		return Joint.NAMES[self.type]


	def draw(self, image):
		# type: (np.ndarray) -> np.ndarray
		"""
		:param image: image on which to draw the joint
		:return: image with the joint
		"""
		image = cv2.circle(
			image, thickness=-1,
			center=self.pos2d,
			radius=self.radius,
			color=self.color,
		)
		return image

	def get_conversion_idx(BASE_NAMES, TARGET_NAMES):
		a = []
		for x in TARGET_NAMES:
			if x in BASE_NAMES:
				a.append(BASE_NAMES.index(x))
			else:
				a.append(-1)
		return a


	def __str__(self):
		visibility = 'visible' if self.visible else 'occluded'
		return "{}|2D:({},{})|3D:({},{},{})|{}".format(self.name,self.x2d,self.y2d,self.x3d,self.y3d,self.z3d,visibility)



	__repr__ = __str__
