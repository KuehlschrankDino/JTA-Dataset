import os
from shutil import copyfile
from path import Path

seq_idx = 500
data_set_root = "../../data/JTA/"
JTA_SEQ_DIR = "../../data/JTA/JTA"
files = [os.path.join(JTA_SEQ_DIR, x, "coords.csv") for x in os.listdir(JTA_SEQ_DIR) if os.path.isfile(os.path.join(JTA_SEQ_DIR,x, "coords.csv"))]
target_anno_dir = Path(os.path.join(data_set_root, "annotations/train/csv"))
if not target_anno_dir.exists():
    target_anno_dir.makedirs()
for file in files:
    vid_id = "{:06d}".format(seq_idx)

    dest = os.path.join(target_anno_dir, "seq_{}.csv".format(vid_id))
    copyfile(file, dest)

    target_dir = Path(os.path.join(data_set_root, "images/train/", "seq_{}".format(seq_idx)))
    if not target_dir.exists():
        target_dir.makedirs()
    source_dir = os.path.dirname(file)
    for img_file in os.listdir(source_dir):
        if(img_file.endswith(".jpeg")):
            src_file = os.path.join(source_dir, img_file)
            tgt_file = os.path.join(target_dir, "{}.jpg".format(img_file.split(".")[0]))
            copyfile(src_file, tgt_file)
    seq_idx += 1