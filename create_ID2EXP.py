import os
from utils import RESOLUTION, expression_conversion

id2exp = {}

id = 0
for video in RESOLUTION:
    path = os.path.join("./plugins/Refer-KITTI/expression", video)
    dirs = os.listdir(path)
    for dir in dirs:
        dir = dir.split(".")[0]
        dir = expression_conversion(dir)
        if dir not in id2exp.values():
            id += 1
            id2exp[id] = dir

with open("ID2EXP.txt", "w", encoding="utf-8") as file:
    file.write("ID2EXP = {" + "\n")
    for key, value in id2exp.items():
        file.write(str(key) + ": " + '"' + value + '",' + "\n")
    file.write("}\n")