import cv2
import os
import sys
from path import Path
import tqdm



JTA_DATASET_ROOT = "../../data/JTA-Data/"
JTA_IMAGES_ROOT = os.path.join(JTA_DATASET_ROOT, "images")
JTA_VIDEO_ROOT = os.path.join(JTA_DATASET_ROOT, "videos")
training_sets = [os.path.join(JTA_VIDEO_ROOT, x) for x in os.listdir(JTA_VIDEO_ROOT) if os.path.isdir(os.path.join(JTA_VIDEO_ROOT, x))]
for i, set_path in enumerate(training_sets):
    videos = [os.path.join(set_path, x) for x in os.listdir(set_path) if x.endswith(".mp4")]
    set = os.path.basename(set_path)

    for vid_num, video_path in enumerate(videos):
        print("Set {} of {} --- Sequence {} of {} in current set [{}].".format(i, len(training_sets), vid_num, len(videos), set))
        seq_name = os.path.basename(video_path).split(".")[0]
        images_out_path = Path(os.path.join(JTA_IMAGES_ROOT, set, seq_name))

        if not images_out_path.exists():
            images_out_path.makedirs()

        vidcap = cv2.VideoCapture(video_path)

        success, image = vidcap.read()
        img_num = 0
        while success:
            cv2.imwrite(os.path.join(images_out_path, "{:06d}.jpeg".format(img_num)) ,image)     # save frame as JPEG file
            success, image = vidcap.read()
            img_num += 1
