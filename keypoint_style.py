import numpy as np


KEYPOINT_SKELTIONS =dict(
       COCO = [(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12)],
       PoseTrack = [(0, 1), (0, 2), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12)],
       CrowdPose = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (12, 13)])


KEYPOINT_NAMES = dict(
    JTA=['head_top', 'head_center', 'neck', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
         'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist', 'spine0', 'spine1', 'spine2','spine3',
          'spine4', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'],
    CrowdPose=['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head_top', 'neck'],
    PoseTrack=['nose', 'neck', 'head_top', 'left_ear', 'right_ear', 'left_shoulder',
                'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'])


class KeypointConverter():

    def __init__(self, target_style, base_style="JTA"):
        self.TARGET_NAMES = KEYPOINT_NAMES[target_style]
        self.BASE_NAMES = KEYPOINT_NAMES[base_style]
        self.SWAPINDEXS, self.DROPINDEXS = self._calulate_conversion_idxs()
        self.TARGET_STYLE = target_style

    def _calulate_conversion_idxs(self):
        """
        Calculates which parts of the base annotation format needs to be dropped and which part needs to be swapped into
        another position.
        :param:
        :return: list of swap positions, list of types to drop
        """
        old_order, drop_idxs = [], []
        for x in self.BASE_NAMES:
            if x in self.TARGET_NAMES:
                old_order.append(x)
            else:
                drop_idxs.append(float(self.BASE_NAMES.index(x)))
        swap_idxs = [old_order.index(x) for x in self.TARGET_NAMES]
        return swap_idxs, drop_idxs

    def get_drop_idxs(self):
        """
        Returns the types of keypoints which are not contained in the target keypoint style.
        :param :
        :return: list of indexes to drop
        """
        return self.DROPINDEXS

    def get_target_kpt_names(self):
        return KEYPOINT_NAMES[self.TARGET_STYLE]

    def get_target_kpt_skeleton(self):
        return KEYPOINT_SKELTIONS[self.TARGET_STYLE]

    def reorder_keypoints(self, data):
        """
        Reorder the keypoints of a pose to align with the target kepyoint format. The base pose should only the same type
        of keyoints as the target pose.
        :param: np.array
        :return: dict
        """
        n = int(np.sum(data[:, 8] > 0.0))
        pose = np.stack([data[self.SWAPINDEXS,3],data[self.SWAPINDEXS,4],data[self.SWAPINDEXS,8]], axis=1)
        return {
            'keypoints': pose.flatten().tolist(),
            'num_keypoints': n,
        }




