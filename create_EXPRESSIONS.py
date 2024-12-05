import os
import json
from utils import *

seq = {}

for video in VIDEOS["test"]:
    path = os.path.join("./plugins/Refer-KITTI/expression-v2", video)
    dirs = os.listdir(path)
    dirs = [dir.split(".")[0] for dir in dirs]
    seq[video] = dirs

with open("seq.json", "w", encoding="utf-8") as data:
    json.dump(seq, data, indent=4)