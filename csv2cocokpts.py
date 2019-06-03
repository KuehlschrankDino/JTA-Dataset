
import numpy as np
from PIL import Image
import os
import json
import argparse
import pandas as pd

from keypoint_style import get_annotation, KEYPOINT_SKELTIONS, KEYPOINT_NAMES





def is_off_screen(x,y, w, h):
    # type: () -> bool
    """
    :return: True if the joint is on screen, False otherwise
    """

    return (0 >= x >= w) or (0 >= y >= h)

def parse_args():
    #todo: add blacklisting of sequences
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoint_style', type=str, default='CrowdPose')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--img_format', type= str, default="jpeg")
    parser.add_argument('--skip_frames', type=int, default=1)
    args = parser.parse_args()
    return args

def set_v_flags(frame_data, w,h):
    '''
    ==========================================================
    NOTE#1: in ###, each keypoint is represented by its (x,y)
    2D location and a visibility flag `v` defined as:
    	- `v=0` ==> not labeled (in which case x=y=0)
    	- `v=1` ==> labeled but not visible
    	- `v=2` ==> labeled and visible
    	- 'v=3' ==> labeled and self occluded
    ========
    '''
    frame_data[:,8][(frame_data[:,8]== 0) & (frame_data[:,9] == 1)] = 3
    frame_data[:, 8][frame_data[:, 8] == 0] = 2
    frame_data[:, 3:9][((0 > frame_data[:, 3]) | (frame_data[:, 3] > w)) | (((0 > frame_data[:, 4]) | (frame_data[:, 4] > h)))] = 0
    return frame_data



def filter_coords(frame_data):
    """
    Filters the coords based on distance on visibility
    :param frame_data:
    :return:
    """

    # filter by visibility flag
    ids = np.unique(frame_data[:, 1])
    occs = np.reshape(frame_data[:, 8], [ids.shape[0], -1])

    vis = np.logical_not(np.all(occs <= 1.0, axis=1))
    vis = np.tile(np.expand_dims(vis, axis=1), [1, occs.shape[1]]).flatten()

    fully_filtered = frame_data[vis, :]

    return fully_filtered


def csv_to_npy(csv_file_path, max_frame = 0):
    df = pd.read_csv(csv_file_path, sep=",")
    df = df.drop(axis=1, labels=["cam_3D_x", "cam_3D_y", "cam_3D_z", "cam_rot_x", "cam_rot_y", "cam_rot_z", "fov"])
    # drop first frame because sometimes it causes issues and also to keep consistency with original dataset
    df.drop(df[df.frame == 0].index,inplace=True)
    df1 = df[df.isna().any(axis=1)]
    if(df1.size > 0):
        nan_frame = df1["frame"].values[0]
        if nan_frame < max_frame:
            max_frame = nan_frame - 1
    if max_frame != 0:
        df.drop(df[df.frame > max_frame].index, inplace=True)
    #discard annotations that exceed number of frames



    return df.to_numpy(dtype=np.float32)


def seq_data_to_dict(coco_dict, data, seq_dir, img_format, w, h, keypoint_style, skip_frames):
    sequence = None
    try:
        sequence = int(seq_dir.split('_')[-1])
    except:
        print('[!] error during conversion.')
        print('\ttry using JSON/CSV files with the original nomenclature.')

    # if sequence in black_list:
    #     print("Skipping because scene is not fitting.")
    # return

    frame_nrs = np.unique(data[:, 0])



    peds_dict = {}
    for n in frame_nrs:

        frame_data = data[data[:, 0] == n, :]
        frame_data = set_v_flags(frame_data, w, h)
        frame_data = filter_coords(frame_data)

        image_id = 10000000000 + sequence * 10000 + (n +1)
        img_file_path = os.path.join(seq_dir, "{}.{}".format(int(n+1), img_format))
        coco_dict['images'].append({
            'license': 4,
            'file_name': img_file_path,
            # 'frame_id': image_id,
            'height': h,
            'width': w,
            'id': image_id
        })

        #iterating over every person id in a single frame
        for p_id in set(frame_data[:, 1]):
            track_id = int(peds_dict.setdefault(p_id, len(peds_dict) + 1))
            annotation = get_annotation(frame_data, p_id, keypoint_style)
            annotation['image_id'] = image_id
            annotation['id'] = image_id * 100000 + track_id
            annotation['category_id'] = 1
            coco_dict['annotations'].append(annotation)


def convert_annos_to_coco_format(csv_files, args):
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
            'keypoints': KEYPOINT_NAMES[args.keypoint_style],
            'skeleton': KEYPOINT_SKELTIONS[args.keypoint_style]
        }]
    }
    img_format = "jpeg"

    for csv_file in csv_files:
        print("▸ converting annotations of {}".format(csv_file))
        # if 1 != int(os.path.dirname(csv_file).split("_")[-1]):
        #     continue
        abs_path = os.path.dirname(csv_file)
        rel_path = os.path.join("images", abs_path.split("/")[-1])
        img_files = [os.path.join(rel_path, x) for x in os.listdir(abs_path) if x.endswith(img_format)]
        img = Image.open(os.path.join(args.dataset_root,img_files[0]))
        w, h = img.size

        data = csv_to_npy(csv_file, len(img_files))
        seq_data_to_dict(coco_dict, data, rel_path, img_format, w, h, args.keypoint_style, args.skip_frames)
        anno_dir = os.path.join(args.dataset_root, "annotations")

    if not os.path.isdir(anno_dir):
        os.mkdir(anno_dir)
    out_file_path = os.path.join(anno_dir, "train_jta.json")

    with open(out_file_path, 'w') as f:
        json.dump(coco_dict, f)
    return

if __name__ == '__main__':
    args = parse_args()
    JTA_SEQ_DIR = os.path.join(args.dataset_root, "images")
    csv_files = [os.path.join(JTA_SEQ_DIR, x, "coords.csv") for x in os.listdir(JTA_SEQ_DIR) if
             os.path.isfile(os.path.join(JTA_SEQ_DIR, x, "coords.csv"))]
    convert_annos_to_coco_format(csv_files, args)