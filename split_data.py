import os
from shutil import copyfile
from path import Path

from pandas import read_csv, DataFrame
import numpy as np
seq_idx = 512
data_set_root = "../../data/JTA/"
JTA_SEQ_DIR = "../../data/JTA/JTA"
files = [os.path.join(JTA_SEQ_DIR, x, "coords.csv") for x in os.listdir(JTA_SEQ_DIR) if os.path.isfile(os.path.join(JTA_SEQ_DIR,x, "coords.csv"))]
target_anno_dir = Path(os.path.join(data_set_root, "annotations/train/"))
if not target_anno_dir.exists():
    target_anno_dir.makedirs()
for file in files:
    if(file == "../../data/JTA/JTA/seq_34/coords.csv"):
        continue
    source_dir = os.path.dirname(file)
    img_files = [x for x in os.listdir(source_dir) if x.endswith(".jpeg")]
    df = read_csv(file, sep=",")
    df = df.drop(axis=1,
                 labels=["cam_3D_x", "cam_3D_y", "cam_3D_z", "cam_rot_x", "cam_rot_y", "cam_rot_z", "fov"])
    df.drop(df[df.frame == 0].index, inplace=True)#drop first frame because sometimes it causes issues and also to keep consistency with original dataset
    df.drop(df[df.frame > len(img_files)].index, inplace=True)#discard annotations that exceed number of frames

    data = df.to_numpy(dtype=np.float32)

    dest = os.path.join(target_anno_dir, "seq_{}.npy".format(seq_idx))
    np.save(dest, data)

    target_dir = Path(os.path.join(data_set_root, "images/train/", "seq_{}".format(seq_idx)))
    if not target_dir.exists():
        target_dir.makedirs()

    for img_file in img_files:
        src_file = os.path.join(source_dir, img_file)
        img_num = int(img_file.split(".")[0])
        if (img_num != 0):#drop first frame because sometimes it causes issues and also to keep consistency with original dataset
            tgt_file = os.path.join(target_dir, "{:06d}.jpg".format(img_num))
            copyfile(src_file, tgt_file)
    seq_idx += 1