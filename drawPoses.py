import json
import click
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

from pose import Pose
import os
MAX_COLORS = 42

LIMBS = [
    (0, 1),
    (1, 2),
    (2, 6),
    (3, 6),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (8, 9),
    (10, 11),
    (11, 12),
    (7, 12),
    (7, 13),
    (13, 14),
    (14, 15),
]
#currently only works with data created with extrac_images
#todo: fix again for drawin MPII data

def get_colors(number_of_colors, cmap_name='rainbow'):
	# type: (int, str) -> List[List[int]]
	"""
	:param number_of_colors: number of colors you want to get
	:param cmap_name: name of the colormap you want to use
	:return: list of 'number_of_colors' colors based on the required color map ('cmap_name')
	"""
	colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, number_of_colors))[:, :-1]*255
	return colors.astype(int).tolist()

def get_center(bb):
    pos = bb[0]
    size = bb[1]
    return (pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0)

def xywh2bbcords(bb):
    return bb[0], (bb[0][0] + bb[1][0], bb[0][1] + bb[1][1])


H1 = "path to the json-File containing the annotations"
#datapath="/home/kaltob/Workspace/Optimization-of-Human-Body-Pose-Estimation-for-Crowd-Applications/code/python/human-pose-estimation.pytorch/data/mpii/images"


datapath = "/home/kaltob/Workspace/Optimization-of-Human-Body-Pose-Estimation-for-Crowd-Applications/code/data/JTA-Data/images/train/"
#datapath = ""
jso = "/home/kaltob/Workspace/human-pose-estimation.pytorch/data/coco/annotations/person_keypoints_val2017.json"
jso1 = "/home/kaltob/Workspace/Optimization-of-Human-Body-Pose-Estimation-for-Crowd-Applications/code/python/human-pose-estimation.pytorch/data/mpii/annot/train.json"
jso2 = "/home/kaltob/Workspace/Optimization-of-Human-Body-Pose-Estimation-for-Crowd-Applications/code/data/JTA-Data/annotations/train/seq_92.json"
@click.command()
@click.option('--json_file_path', type=click.Path(exists=True), prompt='Enter \'json_file_path\'', help=H1)
def main(json_file_path):
    colors = get_colors(number_of_colors=MAX_COLORS, cmap_name='jet')
    with open(json_file_path, 'r') as json_file:
    #with open(jso1, 'r') as json_file:
        data = json.load(json_file)

        for n, x in enumerate(data[::100]):
            pose = np.array(x["joints"]).astype(int)
            visible = np.array(x["joints_vis"])
            path = os.path.join(datapath, x["image"])


            image = cv2.imread(os.path.join(datapath, x["image"]))
            color = colors[int(n) % len(colors)]

            #cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), ( 0, 255, 0) ,thickness=1,)
            bb_xywh = x['bb']
            center = get_center(bb_xywh)
            bb = xywh2bbcords(bb_xywh)

            cv2.rectangle(image, tuple(bb[0]), tuple(bb[1]), (0, 255, 0), thickness=1, )
            image = cv2.circle(
                image,
                thickness=-1,
                center=(int(center[0]), int(center[1])),
                radius=1,
                color=( 0, 255, 0)
            )

            for (j_id_a, j_id_b) in Pose.LIMBS:
                if(visible[j_id_a] + visible[j_id_b] == 2):
                    cv2.line(image,
                         tuple(pose[j_id_a]),
                         tuple(pose[j_id_b]), color=color, thickness=1)

            for n, joint in enumerate(pose):
                if(visible[n]):
                    cv2.circle(
                        image,
                        thickness=-1,
                        center=tuple(joint),
                        radius=2,
                        color=(255, 0, 0)
                    )
                else:
                    cv2.circle(
                        image,
                        thickness=-1,
                        center=tuple(joint),
                        radius=2,
                        color=(0, 0, 255)
                    )


            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



if __name__ == '__main__':
	main()

