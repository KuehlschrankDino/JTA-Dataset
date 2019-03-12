import csv
import os
import numpy as np
import cv2
from joint import Joint
from pose import Pose
#from math import cos, sin, tan, radians
from pandas import read_csv


def visualize(kpts, frame):
    datapath = "../../data/JTA"
    seq_num = 0
    img_path = os.path.join(datapath, "seq_{}".format(seq_num), "{}.jpeg".format(frame))
    image = cv2.imread(img_path)
    for kpt in kpts:
        cv2.circle(image,tuple(kpt[0:2].astype(int)),2,(0,255,0))
    # for (j_id_a, j_id_b) in Pose.LIMBS:
    #     cv2.line(image, tuple(kpts[j_id_a,0:2].astype(int)), tuple(kpts[j_id_b,0:2].astype(int)), (0, 255, 0))
    cv2.imshow("fig", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_all(all, frame):

    datapath = "../../data/JTA"
    seq_num = 0
    img_path = os.path.join(datapath, "seq_{}".format(seq_num), "{}.jpeg".format(frame))
    image = cv2.imread(img_path)
    for kpts in all:
        for kpt in kpts:
            cv2.circle(image,tuple(kpt[0:2].astype(int)),2,(0,255,0))

    cv2.imshow("fig", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def row_to_np(row, person_id):
    return np.array[float(row["frame"]),
                    float(person_id),
                    float(row["joint_type"]),
                    float(row["2D_x"]),
                    float(row["2D_y"]),
                    float(row["3D_x"]),
                    float(row["3D_y"]),
                    float(row["3D_z"]),
                    float(row["occluded"]),
                    float(row["self_occluded"])
                    ]

datapath = "../../data/JTA"
seq_num = 0
csv_anno = os.path.join(datapath, "seq_{}".format(seq_num), "coords.csv")
num_joints = 22


kpts_2d = np.zeros((num_joints,3))
current_frame_kpts = []
vis = [False for i in range(num_joints)]
vid_id = "{:06d}".format(seq_num)
sequence_dump = {"annotations": [],
                "images" :[],
                "categories":[]}
peds_dict = {}
current_frame_anns= []
#todo: write image part of dictionary
data = []

df = read_csv(csv_anno,sep = ",")
df= df.drop(axis=1, labels=["cam_3D_x", "cam_3D_y", "cam_3D_z", "cam_rot_x", "cam_rot_y", "cam_rot_z", "fov"])
data = df.to_numpy(dtype=np.float32)

print("test")# with open(csv_anno, "r") as f:
#     csv_reader = csv.DictReader(f, delimiter = ',')
#     for row in csv_reader:

#
# for frame_number in range(0, 900):
#     image_id = "{}{:04d}".format(vid_id, frame_number)
#     joint_list =[]
#     num_person_in_frame = 0
#     for row in csv_reader:
#         if (int(row["frame"]) == frame_number + 1):#todo: before running check if this is correct
#             continue
#
#         next_ped_id = peds_dict.setdefault(row["pedestrian_id"], len(peds_dict) + 1) #start Person Ids with 0
#         if (current_ped_id != next_ped_id):#create Pose with joint_list if new Person Id
#             if(len(joint_list == num_joints)):
#                 current_ped_id = next_ped_id
#                 pose = Pose(joint_list)
#                 joint_list = []
#                 if not pose.invisible:
#                     annotation = pose.dino_annotation
#                     annotation['image_id'] = image_id
#                     annotation['id'] = "{}{:03d}".format(image_id, num_person_in_frame + 1)
#                     annotation['category_id'] = 1
#                     annotation['track_id'] = current_ped_id
#                     annotation['scores'] = []
#                     num_person_in_frame = num_person_in_frame + 1
#                     sequence_dump['annotations'].append(annotation)
#
#         current_ped_id = next_ped_id
#         joint_list.append(get_Joint_from_csvrow(row, current_ped_id))
# # OrderedDict([('bbox_head', [253, 77, 57, 58]),
# #              ('keypoints', [304, 113, 1, 274.1997681, 137.6983643, 1, 286.9751587, 82.22621918, 1, 0, 0, 0, 0, 0, 0, 243, 143, 1, 301, 151.5, 1, 296, 159.5, 1, 348, 181.5, 1, 311, 124.5, 1, 366, 126.5, 1, 227, 261, 1, 267, 263, 1, 215, 338.5, 1, 312, 358.5, 1, 0, 0, 0, 0, 0, 0]),
# #              ('track_id', 0), ('image_id', 20000010023), ('bbox', [192.35, 40.785152056999976, 196.29999999999998, 359.15591506600003]),
# #              ('scores', []),
# #              ('category_id', 1), ('id', 2000001002300)])




# with open(csv_anno, "r") as f:
#     csv_reader = csv.DictReader(f, delimiter = ',')
#
#     current_frame_idx = 0
#
#     for row in csv_reader:
#         if (int(row["frame"]) == 0):
#             #first frame can contain errors so better exclude it...we have enought frames anyway ;)
#             continue
#         v_flag = evaluate_visibility(row["occluded"]=="1", row["self_occluded"] == "1")
#         kpts_2d[int(row["joint_type"]),] = np.array([float(row["2D_x"]), float(row["2D_y"]), v_flag])
#         Joint()
#         if(int(row["frame"]) > current_frame_idx):
#             current_frame_idx = int(row["frame"])
#             visualize_all(current_frame_kpts, int(row["frame"]) -1)
#             current_frame_kpts = []
#             image_id = "{}{:04d}".format(vid_id, current_frame_idx)
#             sequence_dump["annotations"].extend(current_frame_anns)
#         if(row["joint_type"] == "21"):
#                 current_frame_kpts.append(kpts_2d)
#
#                 current_frame_anns.append({"bbox_head": [],
#                                          "keypoints": kpts_2d.flatten.tolist,
#                                          "track_id": peds_dict.setdefault(row["pedestrian_id"], len(peds_dict) + 1),
#                                          "id": "{}{:03d}".format(image_id, len(current_frame_anns) + 1),
#                                          "image_id": image_id,
#                                          "bbox": get_bb_from_kpts(kpts_2d)})
#                 kpts_2d = np.zeros(kpts_2d.shape)
#
# # OrderedDict([('bbox_head', [253, 77, 57, 58]),
# #              ('keypoints', [304, 113, 1, 274.1997681, 137.6983643, 1, 286.9751587, 82.22621918, 1, 0, 0, 0, 0, 0, 0, 243, 143, 1, 301, 151.5, 1, 296, 159.5, 1, 348, 181.5, 1, 311, 124.5, 1, 366, 126.5, 1, 227, 261, 1, 267, 263, 1, 215, 338.5, 1, 312, 358.5, 1, 0, 0, 0, 0, 0, 0]),
# #              ('track_id', 0), ('image_id', 20000010023), ('bbox', [192.35, 40.785152056999976, 196.29999999999998, 359.15591506600003]),
# #              ('scores', []),
# #              ('category_id', 1), ('id', 2000001002300)])