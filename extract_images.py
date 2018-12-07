import json
import numpy as np
import os
import click
import imageio

from joint import Joint
from path import Path
from pose import Pose


def get_pose(frame_data, person_id):
	# type: (np.ndarray, int) -> Pose
	"""
	:param frame_data: data of the current frame
	:param person_id: person identifier
	:return: list of joints in the current frame with the required person ID
	"""
	pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
	pose.sort(key=(lambda j: j.type))
	return pose


def bbox_2d(list, cam_distance):
    # type: () -> List[int]
    """
    :return: bounding box around the pose in format [x_min, y_min, width, height]
        - x_min = x of the top left corner of the bounding box
        - y_min = y of the top left corner of the bounding box
    """
    #calculate margin based on cam distance
    margin = int((1.0 - (cam_distance / 100.0)) * 20)
    x_min = int(np.min([j[0] for j in list]))
    y_min = int(np.min([j[1] for j in list]))
    x_max = int(np.max([j[0] for j in list]))
    y_max = int(np.max([j[1] for j in list]))

    return ((x_min-margin, y_min-margin), (x_max - x_min +  2 * margin, y_max - y_min + 2* margin))

def extract(in_mp4_path, json_path, out_dir_path_img, out_dir_path_anno, first_frame, skip_frames, img_set):
    in_mp4_path = os.path.join(in_mp4_path, img_set)
    json_path = os.path.join(json_path, img_set)
    out_dir_path_img = Path(os.path.join(out_dir_path_img, img_set))
    out_dir_path_anno = Path(out_dir_path_anno)
    out_data = []
    # create dirs if they do not already exist
    if not out_dir_path_img.exists():
        out_dir_path_img.makedirs()
    if not out_dir_path_anno.exists():
        out_dir_path_anno.makedirs()

    if (os.path.isdir(in_mp4_path)):
        for filename in os.listdir(in_mp4_path):
            print("Currently processing " + filename)
            in_mp4_file = os.path.join(in_mp4_path, filename)
            reader = imageio.get_reader(in_mp4_file)
            name = os.path.splitext(filename)[0]
            json_file_path = os.path.join(json_path, name + ".json")

            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                data = np.array(data)

            for frame_number, image in enumerate(reader):
                if (frame_number % skip_frames == 0):  # skip every nth frame
                    image_file_name = name + "_" + str(frame_number) + ".jpg"
                    frame_data = data[data[:, 0] == frame_number + 1]

                    imageio.imwrite(os.path.join(out_dir_path_img, image_file_name), image)

                    # extract pose and create bounding box for every person
                    for p_id in set(frame_data[:, 1]):
                        pose = get_pose(frame_data=frame_data, person_id=p_id)

                        visible, joints, soc, oc = [], [], [], []
                        line = {}
                        min_distance = 9999.0
                        for joint in pose:
                            min_distance = min(joint.cam_distance, min_distance)
                            visible.append(int(joint.visible and joint.is_on_screen))
                            soc.append(joint.soc)
                            oc.append(joint.occ)
                            joints.append([joint.pos2d[0], joint.pos2d[1]])

                        # discard pose if less than 2 joints are visible, or person is to far away
                        if (sum(visible) > 4 and min_distance < 40.0):
                            line['joints_vis'] = visible
                            line['joints'] = joints
                            line['image'] = image_file_name
                            line['occluded'] = oc
                            line['self_occluded'] = soc
                            line['bb'] = bbox_2d(joints, min_distance)
                            out_data.append(line)

        with open(os.path.join(out_dir_path_anno, img_set + ".json"), 'w') as outfile:
            json.dump(out_data, outfile)

#todo: check if these descriptions are correct
H1 = 'path of the video or directory of videos from which you want to extract the frames'
H2 = 'directory where you want to save the extracted frames'
H3 = 'number from which to start counting the video frames; DEFAULT = 1'
H4 = 'the format to use to save the images/frames; DEFAULT = jpg'
H5 = 'the frames that are skipped between extracting'
H6 = 'Path to Annotation JSON-File'
H7 = 'test, train or val set'


#todo: clean up options
@click.command()
@click.option('--in_mp4_path', type=click.Path(exists=True), prompt='Enter \'in_mp4_file_path\'', help=H1)
@click.option('--out_dir_path_img', type=click.Path(), prompt='Enter \'out_dir_path\'', help=H2)
@click.option('--out_dir_path_anno', type=click.Path(), prompt='Enter \'out_dir_path\'', help="")
@click.option('--first_frame', type=int, default=1, help=H3)
@click.option('--skip_frames', type=int, default=30, help=H5)
@click.option('--json_path', type=click.Path(exists=True), prompt='Enter \'json_file_path\'', help=H6)
def main(in_mp4_path, json_path, out_dir_path_img, out_dir_path_anno, first_frame, skip_frames):
    img_set = ["val"]
    for set in img_set:
        extract(in_mp4_path, json_path, out_dir_path_img, out_dir_path_anno, first_frame, skip_frames, set)



if __name__ == '__main__':
    main()




