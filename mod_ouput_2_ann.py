import csv
import os
import numpy as np
import cv2
#from math import cos, sin, tan, radians
from numpy import sin, cos, tan, radians

from pose import Pose
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
    seq_num = 1
    img_path = os.path.join(datapath, "seq_{}".format(seq_num), "{}.jpeg".format(frame))
    image = cv2.imread(img_path)
    for kpts in all:
        for kpt in kpts:
            cv2.circle(image,tuple(kpt[0:2].astype(int)),2,(0,255,0))

    cv2.imshow("fig", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


datapath = "../../data/JTA"
seq_num = 0
csv_anno = os.path.join(datapath, "seq_{}".format(seq_num), "coords.csv")
num_joints = 22

kpts_3d = np.zeros((num_joints,3))
kpts_2d = np.zeros((num_joints,2))
all = []
vis = [False for i in range(num_joints)]
with open(csv_anno, "r") as f:
    csv_reader = csv.DictReader(f, delimiter = ',')

    for row in csv_reader:
        kpts_3d[int(row["joint_type"]),] = np.array([float(row["3D_x"]), float(row["3D_y"]), float(row["3D_z"])])
        kpts_2d[int(row["joint_type"]),] = np.array([float(row["2D_x"]), float(row["2D_y"])])
        if(int(row["frame"]) == 1):
            visualize_all(all, int(row["frame"]))
        if(row["joint_type"] == "21"):
                # projMat = get_cam_projectionMatrix(float(row["cam_rot_x"]), float(row["cam_rot_y"]), float(row["cam_rot_z"]))
                # cam_pos = np.array([float(row[" cam_3D_x"]), float(row["cam_3D_y"]), float(row["cam_3D_z"])])
                # n_kpts = convert_3d_to_2d(kpts, cam_pos, projMat, float(row[" fov"]))
                # visualize(kpts_2d, int(row["frame"]))
                all.append(kpts_2d)
                kpts_2d = np.zeros(kpts_2d.shape)
