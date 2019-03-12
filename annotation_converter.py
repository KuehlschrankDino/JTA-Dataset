# -*- coding: utf-8 -*-
# ---------------------

import json
import sys
import os.path as osp
import click
import imageio
import numpy as np
from path import Path
from pandas import read_csv

from joint import Joint
from pose import Pose


imageio.plugins.ffmpeg.download()
MAX_COLORS = 42

# check python version ##the world is not ready for f-strings yet
#assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'


def get_pose(frame_data, person_id):
    # type: (np.ndarray, int) -> Pose
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :return: list of joints in the current frame with the required person ID
    """
    pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
    pose.sort(key=(lambda j: j.type))
    return Pose(pose)
H1 = 'style of the annotation format'
H2 ='style of the keypoint format'
H3 = 'path of the output directory'
H4 = 'path of the annotation directory'


@click.command()
@click.option('--annotation_style', type=click.Choice(['coco', 'PoseTrack']), prompt='Choose \'annotation_style\'', help=H1)
@click.option('--keypoint_style', type=click.Choice(['JTA', 'CrowdPose', 'PoseTrack']), prompt='Choose \'keypoint_style\'', help=H2)
@click.option('--out_dir_path', type=click.Path(), prompt='Enter \'out_dir_path\'', help=H3)
@click.option('--dataset_root', type=click.Path(), prompt='Enter \'annotations_root\'', help=H3)
def main(annotation_style, keypoint_style, out_dir_path):
    # type: (str) -> None
    """
    Script for annotation conversion (from JTA format to COCO format)
    """

    out_dir_path = Path(out_dir_path).abspath()
    if not out_dir_path.exists():
        out_dir_path.makedirs()

    for dir in Path('annotations').dirs():
        out_subdir_path = out_dir_path / dir.basename()
        if not out_subdir_path.exists():
            out_subdir_path.makedirs()
        print(f'▸ converting \'{dir.basename()}\' set')
        for anno in dir.files():

            if anno.endswith('.json'):
                with open(anno, 'r') as json_file:
                    data = json.load(json_file)
                    data = np.array(data)
            elif anno.endswith('.csv'):
                df = read_csv(anno, sep=",")
                df = df.drop(axis=1,
                             labels=["cam_3D_x", "cam_3D_y", "cam_3D_z", "cam_rot_x", "cam_rot_y", "cam_rot_z", "fov"])
                data = df.to_numpy(dtype=np.float32)

            print("▸ converting annotations of".format(Path(anno).abspath()))
            #print(f'▸ converting annotations of \'{Path(anno).abspath()}\'')

            # getting sequence number from `anno`
            sequence = None
            try:
                sequence = int(Path(anno).basename().split('_')[1].split('.')[0])
            except:
                print('[!] error during conversion.')
                print('\ttry using JSON/CSV files with the original nomenclature.')

            coco_dict = {
                'info': {
                    'description': f'JTA 2018 Dataset - Sequence #{sequence}',
                    'url': 'http://aimagelab.ing.unimore.it/jta',
                    'version': '1.0',
                    'year': 2018,
                    'contributor': 'AImage Lab',
                    'date_created': '2018/01/28',
                },
                'licences': [{
                    'url': 'http://creativecommons.org/licenses/by-nc/2.0',
                    'id': 2,
                    'name': 'Attribution-NonCommercial License'
                }],
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

            for frame_number in range(0, 900):

                image_id = sequence * 1000 + (frame_number + 1)
                coco_dict['images'].append({
                    'license': 4,
                    'file_name': f'{frame_number + 1}.jpg',
                    'height': 1080,
                    'width': 1920,
                    'date_captured': '2018-01-28 00:00:00',
                    'id': image_id
                })

                # NOTE: frame #0 does NOT exists: first frame is #1
                frame_data = data[data[:, 0] == frame_number + 1]  # type: np.ndarray

                for p_id in set(frame_data[:, 1]):#todo check that the p_ids are not really large number
                    pose = get_pose(frame_data=frame_data, person_id=p_id)

                    # ignore the "invisible" poses
                    # (invisible pose = pose of which I do not see any joint)
                    if pose.invisible:
                        continue

                    annotation = pose.coco_annotation
                    annotation['image_id'] = image_id
                    annotation['id'] = image_id * 100000 + int(p_id)
                    annotation['category_id'] = 1
                    coco_dict['annotations'].append(annotation)

                #print(f'\r▸ progress: {100 * (frame_number / 899):6.2f}%', end='')

            out_file_path =osp.join(out_subdir_path, "seq_{}_{}".format(sequence,annotation_style) )
            # out_file_path = out_subdir_path / f'seq_{sequence}.coco.json'
            with open(out_file_path, 'w') as f:
                json.dump(coco_dict, f)


if __name__ == '__main__':
    main()
