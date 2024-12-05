import os
import copy
import json
from utils import *
from utils import WORDS_MAPPING

labels = {}

for video in RESOLUTION.keys():
    labels[video] = {}
    
    path = os.path.join('./plugins/Refer-KITTI/expression', video)
    files = os.listdir(path)   
    for file in files:
        with open(os.path.join(path, file), "r", encoding="utf-8") as data_file:
            data = json.load(data_file)
            label = data["label"]
            sentence = data["sentence"]
            expression = expression_conversion(sentence)

            for frame_id, object_ids in label.items():
                str_frame_id = copy.deepcopy(frame_id)
                frame_id = int(frame_id)
                for object_id in object_ids:
                    str_object_id = copy.deepcopy(object_id)
                    object_id = int(object_id)
                    if object_id not in labels[video].keys():
                        labels[video][int(object_id)] = {}
                    labels[video] = {key: labels[video][key] for key in sorted(labels[video].keys())}

                    if frame_id not in labels[video][object_id].keys():
                        labels[video][object_id][frame_id] = {}
                    labels[video][object_id] = {key: labels[video][object_id][key] for key in sorted(labels[video][object_id].keys())}

                    if "category" not in labels[video][object_id][frame_id].keys():
                        labels[video][object_id][frame_id]["category"] = []
                        for pronoun in WORDS_MAPPING.keys():
                            if pronoun in expression.split(" "):
                                if WORDS_MAPPING[pronoun] == "car":
                                    labels[video][object_id][frame_id]["category"].append("car")
                                else:
                                    labels[video][object_id][frame_id]["category"].append("pedestrian")
                                break
                    
                    if "expression_new" not in labels[video][object_id][frame_id].keys():
                        labels[video][object_id][frame_id]["expression_new"] = []
                    if expression not in labels[video][object_id][frame_id]["expression_new"]:
                        labels[video][object_id][frame_id]["expression_new"].append(expression)
                    labels[video][object_id][frame_id]["expression_new"].sort()

                    labels[video][object_id][frame_id]["bbox"] = [0.0, 0.0, 0.0, 0.0]
                    
                    bbox_path = os.path.join('./plugins/Refer-KITTI/KITTI/labels_with_ids/image_02', video, str_frame_id.zfill(6)+".txt")
                    with open(bbox_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            line = line.strip().split()
                            if int(line[1]) == object_id:
                                labels[video][object_id][frame_id]["bbox"] = [float(x) for x in line[2:6]]
                                break

with open('./plugins/Refer-KITTI_labels.json', "w", encoding="utf-8") as file:
    json.dump(labels, file, ensure_ascii=False, indent=4)

print("successful!")