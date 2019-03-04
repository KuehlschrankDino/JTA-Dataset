import csv
import os
import numpy as np
import cv2
#from math import cos, sin, tan, radians
from numpy import sin, cos, tan, radians

from pose import Pose
def visualize(kpts, frame):
    datapath = "../../data/JTA"
    seq_num = 1
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

def get_cam_projectionMatrix(alpha, beta, gamma):
    alpha = radians(alpha)
    beta = radians(beta)
    gamma = radians(gamma)

    return np.array([[cos(beta)*cos(gamma),
                      -cos(alpha)*sin(gamma)+sin(alpha)*sin(beta)*cos(gamma),
                      sin(alpha)*sin(gamma) + cos(alpha)*sin(beta) * cos(gamma)],

                    [cos(beta)*sin(gamma),
                     cos(alpha)*cos(gamma)+sin(alpha)*sin(beta)*sin(gamma),
                     -sin(alpha)*cos(gamma)+cos(alpha)*sin(beta)*sin(gamma)],

                    [-sin(beta), sin(alpha)*cos(beta), cos(alpha)*cos(beta)]])


def convert_3d_to_2d(kpts, cam_pos, projMatrix, fov, width=1920, height=1080):
    x = radians(fov) / 2.0
    f = height / 2.0 * (1/ tan(x))
    kpts_2d = np.zeros(kpts.shape)
    for i, kpt in enumerate(kpts):
        hmmm =  np.dot(kpt - cam_pos, projMatrix)
        hmmm2 = np.dot(projMatrix, kpt - cam_pos)
        kpts_2d[i,] = np.dot(kpt - cam_pos, projMatrix)
        # kpts_2d[i,] = np.matmul(projMatrix, kpt - cam_pos)
        

        kpts_2d[i,] = [f*(kpt[0] / kpt[1]) + width / 2.0, height / 2.0 - f* kpt[2] / kpt[1], 1.0]
    return kpts_2d


datapath = "../../data/JTA"
seq_num = 1
csv_anno = os.path.join(datapath, "seq_{}".format(seq_num), "coords.csv")

kpts = np.zeros((22,3))
all = []
vis = [False for i in range(22)]
with open(csv_anno, "r") as f:
    csv_reader = csv.DictReader(f, delimiter = ',')

    for row in csv_reader:
        kpts[int(row[" joint_type"]),] = np.array([float(row[" 3D_x"]), float(row[" 3D_y"]), float(row[" 3D_z"])])
        vis[int(row[" joint_type"])] = (int(row[" occluded"]) == 0)
        if(row[" joint_type"] == "21" and any(vis)):
                projMat = get_cam_projectionMatrix(float(row[" cam_rot_x"]), float(row[" cam_rot_y"]), float(row[" cam_rot_z"]))
                cam_pos = np.array([float(row[" cam_3D_x"]), float(row[" cam_3D_y"]), float(row[" cam_3D_z"])])
                n_kpts = convert_3d_to_2d(kpts, cam_pos, projMat, float(row[" fov"]))
                #visualize(kpts, int(row["frame"]))
                all.append(n_kpts)
                kpts = np.zeros((22,3))
        if(int(row["frame"]) == 2):
            visualize_all(all, int(row["frame"]))