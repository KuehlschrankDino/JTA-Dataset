import os
from shutil import copyfile



files = [os.path.join(x, "coords.csv") for x in os.listdir() if os.path.isfile(os.path.join(x, "coords.csv"))]
for file in files:
    dest = os.path.join("../annotations/train/csv", "{}.csv".format(os.path.dirname(file)))
    copyfile(file, dest)