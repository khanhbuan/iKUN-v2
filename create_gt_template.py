import os
import json
from utils import *

for video in VIDEOS["test"]:
    path = os.path.join("./plugins/Refer-KITTI/expression", video)
    H, W = RESOLUTION[video]
    dirs = os.listdir(path)
    for dir in dirs:
        with open(os.path.join(path, dir), "r", encoding='utf-8') as file:
            data = json.load(file)["label"]
            os.makedirs(os.path.join('./plugins/Refer-KITTI/gt_template', video, dir.split('.')[0]))
            with open(os.path.join('./plugins/Refer-KITTI/gt_template', video, dir.split('.')[0], "gt.txt"), "w", encoding="utf-8") as des_file:
                for frame_id, object_ids in data.items():
                    object_ids.sort()
                    bbox_path = os.path.join('./plugins/Refer-KITTI/KITTI/labels_with_ids/image_02', video, frame_id.zfill(6)+".txt")
                    with open(bbox_path, "r", encoding="utf-8") as bbox_file:
                        for line in bbox_file:
                            line = line.strip().split()
                            if int(line[1]) in object_ids:
                                sampl = str(frame_id) + " " + str(line[1])
                                sampl = sampl + " " + str(float(line[2])*W) + " " + str(float(line[3]) * H)
                                sampl = sampl + " " + str(float(line[4])*W) + " " + str(float(line[5]) * H)
                                sampl = sampl + " 1 1 1"
                                des_file.write(sampl + "\n")