
import numpy as np
from PIL import Image
import os
import json
json.encoder.FLOAT_REPR = lambda x: format(x, '.2f')
import argparse
import pandas as pd

from keypoint_style import get_annotation, KEYPOINT_SKELTIONS, KEYPOINT_NAMES


frame_offsets = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 1,
    10: 2,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 2,
    19: 1,
    20: -1,
    21: -1,
    22: -1,
    23: 1,
    24: 1,
    25: 2
}

black_list = [19, 20, 21, 22]

def parse_args():
    parser = argparse.ArgumentParser()
    #todo: add option to combine with existing dataset in fabbri-json format.
    #todo: add crowdindex calculation.
    parser.add_argument('--keypoint_style', type=str, default='CrowdPose')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--img_format', type= str, default="jpeg")
    parser.add_argument('--skip_frames', type=int, default=1)
    args = parser.parse_args()
    return args

def set_v_flags(frame_data, w,h):
    '''
    ==========================================================
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


def seq_data_to_dict(coco_dict, data, seq_dir, img_format, w, h, keypoint_style, skip_frames,dataset_root):
    sequence = None
    try:
        sequence = int(seq_dir.split('_')[-1])
    except:
        print('[!] error during conversion.')
        print('\ttry using CSV files with the original nomenclature.')

    frame_nrs = np.unique(data[:, 0])

    peds_dict = {}
    for n in frame_nrs:

        frame_data = data[data[:, 0] == n, :]
        frame_data = set_v_flags(frame_data, w, h)
        frame_data = filter_coords(frame_data)
        n_offset = int(n + frame_offsets[sequence])
        image_id = 10000000000 + sequence * 10000 + (n_offset)
        img_file_path = os.path.join(seq_dir, "{}.{}".format(n_offset, img_format))

        if not os.path.isfile(os.path.join(dataset_root, img_file_path)):
            print("Skipping Frame: {}. No image found in: {}".format(n_offset, img_file_path))
            continue

        coco_dict['images'].append({
            'license': 4,
            'file_name': img_file_path,
            # 'frame_id': image_id,
            'height': h,
            'width': w,
            'id': int(image_id)
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
    img_format = args.img_format.split(".")[-1]

    for csv_file in csv_files:
        print("▸ converting annotations of {}".format(csv_file))
        seq_n = int(os.path.dirname(csv_file).split("_")[-1])
        # if seq_n != 11:
        #     print("Skipping Sequence {} because it is blacklisted.".format(seq_n))
        #     continue
        if seq_n in black_list:
            print("Skipping Sequence {} because it is blacklisted.".format(seq_n))
            continue
        abs_path = os.path.dirname(csv_file)
        rel_path = os.path.join("images", abs_path.split("/")[-1])
        img_files = [os.path.join(rel_path, x) for x in os.listdir(abs_path) if x.endswith(img_format)]
        img = Image.open(os.path.join(args.dataset_root,img_files[0]))
        w, h = img.size

        data = csv_to_npy(csv_file, len(img_files))
        seq_data_to_dict(coco_dict, data, rel_path, img_format, w, h, args.keypoint_style, args.skip_frames, args.dataset_root)
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