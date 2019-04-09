# -*- coding: utf-8 -*-
# ---------------------

import json
import sys
import os.path as osp
import click
import cv2
import numpy as np
from path import Path
from pandas import read_csv

from joint import Joint
from pose import Pose



MAX_COLORS = 42
NUMBER_OF_FRAMES_TO_SKIP = 1
# check python version ##the world is not ready for f-strings yet
#assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'


def get_keypoint_names_from_style(keypoint_style):
    if (keypoint_style == "JTA"):
        return Joint.NAMES
    if (keypoint_style == "CrowdPose"):
        return Joint.NAMES_CROWDPOSE
    if (keypoint_style == "PoseTrack"):
        return Joint.NAMES_POSETRACK

def get_skeleton_from_keypoint_style(keypoint_style):
    if (keypoint_style == "JTA"):
        return Pose.SKELETON
    if (keypoint_style == "CrowdPose"):
        return Pose.SKELETON_CROWD_POSE


def get_pose(frame_data, person_id, keypoint_style):
    # type: (np.ndarray, int) -> Pose
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :return: list of joints in the current frame with the required person ID
    """
    pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
    pose.sort(key=(lambda j: j.type))
    return Pose(pose, keypoint_style)

H2 ='style of the keypoint format'
H3 = 'path of the output directory'
H4 = 'path of the annotation directory'


@click.command()
@click.option('--keypoint_style', type=click.Choice(['JTA', 'CrowdPose', 'PoseTrack']), prompt='Choose \'keypoint_style\'', help=H2)
@click.option('--out_dir_path', type=click.Path(), prompt='Enter \'out_dir_path\'', help=H3)
@click.option('--dataset_root', type=click.Path(), prompt='Enter \'annotations_root\'', help=H3)
def main(dataset_root, keypoint_style, out_dir_path):
    # type: (str) -> None
    """
    Script for annotation conversion (from JTA format to COCO format)
    """

    out_dir_path = Path(out_dir_path).abspath()
    if not out_dir_path.exists():
        out_dir_path.makedirs()
    annotations_path = osp.join(dataset_root, 'annotations')
    for dir in Path(annotations_path).dirs():
        # if "test" !=dir.basename():
        #     continue
        out_subdir_path = out_dir_path / dir.basename()
        if not out_subdir_path.exists():
            out_subdir_path.makedirs()
        print("'▸ converting {} set".format(dir.basename()))

        for anno in dir.files():
            # if("seq_0" != osp.basename(anno).split(".")[0]):
            #     continue
            if anno.endswith('.json'):
                with open(anno, 'r') as json_file:
                    data = json.load(json_file)
                    data = np.array(data)
            elif anno.endswith('.npy'):
                data = np.load(anno)

            print("▸ converting annotations of {}".format(Path(anno).abspath()))


            # getting sequence number from `anno`
            sequence = None
            try:
                sequence = int(Path(anno).basename().split('_')[1].split('.')[0])
            except:
                print('[!] error during conversion.')
                print('\ttry using JSON/CSV files with the original nomenclature.')

            coco_dict = {
                'images': [],
                'annotations': [],
                'categories': [{
                    'supercategory': 'person',
                    'id': 1,
                    'name': 'person',
                    'keypoints': get_keypoint_names_from_style(keypoint_style),
                    'skeleton': get_skeleton_from_keypoint_style(keypoint_style)
                }]
            }
            peds_dict = {}
            vid_id = "{:06d}".format(sequence)
            for frame_number in range(0, 1800):

                if(frame_number % NUMBER_OF_FRAMES_TO_SKIP != 0):
                    continue
                image_id = 10000000000 + sequence * 10000 + (frame_number + 1)

                # image_id = sequence * 1000 + (frame_number + 1) JTA BEFORE

                img_path = osp.join("images", dir.basename(),
                                    osp.basename(anno).split(".")[0],
                                    "{:06d}.jpg".format(frame_number + 1))

                if not osp.isfile(osp.join(dataset_root, img_path)):
                    print("{} does not exist...".format(osp.join(dataset_root, img_path)))
                    break

                coco_dict['images'].append({
                    'license': 4,
                    'file_name': img_path,    #f'{frame_number + 1}.jpg',
                    'vid_id': vid_id,
                    'frame_id': image_id,
                    'height': 1080,
                    'width': 1920,
                    'id': image_id
                })

                # NOTE: frame #0 does NOT exists: first frame is #1
                frame_data = data[data[:, 0] == frame_number + 1]  # type: np.ndarray

                # img = cv2.imread(osp.join(dataset_root, img_path))
                for p_id in set(frame_data[:, 1]):#todo check that the p_ids are not really large number
                    pose = get_pose(frame_data=frame_data, person_id=p_id, keypoint_style=keypoint_style)

                    #ignore poses which are nor or only barely on screen
                    if pose.visible_and_onscreen_atleast_n :#and not pose.too_far_from_camera:


                        # img = pose.draw(img, [0, 255, 0])

                        track_id = int(peds_dict.setdefault(p_id, len(peds_dict) + 1))
                        annotation = pose.dino_annotation
                        annotation['image_id'] = image_id
                        annotation['track_id'] = track_id
                        annotation['id'] = image_id * 100000 + track_id
                        annotation['category_id'] = 1
                        coco_dict['annotations'].append(annotation)

                # cv2.imshow(img_path, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            out_file_path =osp.join(out_subdir_path, "seq_{}.json".format(vid_id))
            with open(out_file_path, 'w') as f:
                json.dump(coco_dict, f)

if __name__ == '__main__':
    main()
