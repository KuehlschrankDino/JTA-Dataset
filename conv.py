# -*- coding: utf-8 -*-
# ---------------------

import json
import sys
import os
import click
import imageio
import numpy as np
from path import Path
from pandas import read_csv

from joint import Joint
from pose import Pose



MAX_COLORS = 42

# check python version ##the world is not ready for f-strings yet
#assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'


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
def main(keypoint_style, out_dir_path, dataset_root):
    # type: (str) -> None
    """
    Script for annotation conversion (from JTA format to COCO format)
    """

    out_dir_path = Path(out_dir_path).abspath()
    if not out_dir_path.exists():
        out_dir_path.makedirs()

    anno_files = [os.path.join(dataset_root, x, "coords.csv") for x in os.listdir(dataset_root) if os.path.isfile(os.path.join(dataset_root,x, "coords.csv"))]


    for anno in anno_files:

        df = read_csv(anno, sep=",")
        df = df.drop(axis=1,
                     labels=["cam_3D_x", "cam_3D_y", "cam_3D_z", "cam_rot_x", "cam_rot_y", "cam_rot_z", "fov"])
        data = df.to_numpy(dtype=np.float32)

        print("▸ converting annotations of {}".format(Path(anno).abspath()))
        #print(f'▸ converting annotations of \'{Path(anno).abspath()}\'')

        # getting sequence number from `anno`
        sequence = None
        try:
            sequence = int(Path(anno).dirname().split('_')[1].split('.')[0])
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
                'keypoints': Joint.NAMES,
                'skeleton': Pose.SKELETON
            }]
        }
        peds_dict = {}
        vid_id = "{:06d}".format(sequence)
        for frame_number in range(0, 900):

            image_id = 10000000000 + sequence * 10000 + (frame_number + 1)

            coco_dict['images'].append({
                'license': 4,
                'file_name': os.path.join(Path(anno).dirname(),"{}.jpeg".format(frame_number + 1)),
                'vid_id': vid_id,
                'frame_id': image_id,
                'height': 1080,
                'width': 1920,
                'id': image_id
            })

            # NOTE: frame #0 does NOT exists: first frame is #1
            frame_data = data[data[:, 0] == frame_number + 1]  # type: np.ndarray

            for p_id in set(frame_data[:, 1]):#todo check that the p_ids are not really large number
                pose = get_pose(frame_data=frame_data, person_id=p_id, keypoint_style = keypoint_style)

                # ignore the "invisible" poses
                # (invisible pose = pose of which I do not see any joint)
                if pose.invisible:
                    continue
                track_id = int(peds_dict.setdefault(p_id, len(peds_dict) + 1))
                annotation = pose.dino_annotation
                annotation['image_id'] = image_id
                annotation['track_id'] = track_id
                annotation['id'] = image_id * 100000 + track_id
                annotation['category_id'] = 1
                coco_dict['annotations'].append(annotation)

            #print(f'\r▸ progress: {100 * (frame_number / 899):6.2f}%', end='')

        out_file_path =os.path.join(out_dir_path, "seq_{}.json".format(vid_id) )
        # out_file_path = out_subdir_path / f'seq_{sequence}.coco.json'
        with open(out_file_path, 'w') as f:
            json.dump(coco_dict, f)


if __name__ == '__main__':
    main()
