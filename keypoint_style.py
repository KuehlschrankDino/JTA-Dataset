import numpy as np

conversion_idx = None


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


def get_conversion_idx(BASE_NAMES, TARGET_NAMES):
    a = []
    for x in TARGET_NAMES:
        if x in BASE_NAMES:
            a.append(BASE_NAMES.index(x))
        else:
            a.append(999)
    return a


def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)

    z = [x for _, x in sorted(zipped_pairs)]

    return z


def bbox_2d(pose):
    # type: () -> List[int]
    """
    :return: bounding box around the pose in format [x_min, y_min, width, height]
        - x_min = x of the top left corner of the bounding box
        - y_min = y of the top left corner of the bounding box
    """
    x_min = int(np.min(pose[:,0]))
    y_min = int(np.min(pose[:,1]))
    x_max = int(np.max(pose[:,0]))
    y_max = int(np.max(pose[:,1]))
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def get_annotation(frame_data, person_id, keypoint_style, base_keypoint_Style="JTA"):
    # type: (np.ndarray, int) -> Pose
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :param keypoint_style: keypoint format in which the pose should be converted to
    :return: list of joints in the current frame of the required person ID
    """
    global conversion_idx
    if (conversion_idx is None):
        conversion_idx = get_conversion_idx(KEYPOINT_NAMES[keypoint_style], KEYPOINT_NAMES[base_keypoint_Style])
    pose = [[int(j[3]), int(j[4]), int(j[8])] for j in frame_data[frame_data[:, 1] == person_id]]
    pose = sort_list(pose, conversion_idx)
    pose = np.array(pose)
    bbox = bbox_2d(pose)

    pose = pose[:len(KEYPOINT_NAMES[keypoint_style])]

    annotation = {
        'bbox': bbox,
        'keypoints': pose.flatten().tolist(),
        'num_keypoints': len(conversion_idx),
        'iscrowd': 0,
        'area': bbox[2] * bbox[3]
    }
    return annotation

