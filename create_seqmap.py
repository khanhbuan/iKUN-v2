import os
from utils import *

with open("./plugins/seqmap.txt", "w", encoding="utf-8") as file:
    for video in VIDEOS["test"]:
        path = os.path.join("./plugins/Refer-KITTI/expression", video)
        exps = os.listdir(path)
        exps = [video + "+" + exp.split('.')[0] + "\n" for exp in exps]
        for exp in exps:
            file.write(exp)