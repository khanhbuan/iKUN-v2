import os
import cv2
import numpy as np
from utils import *
import pandas as pd
import argparse

def draw_NeuralSORT(video, exp):
    tracks_1 = np.loadtxt(os.path.join('./plugins/NeuralSORT', video, 'car', 'predict.txt'), delimiter=',')
    if len(tracks_1.shape) == 2:
        tracks = tracks_1
        max_obj_id = max(tracks_1[:, 1])
    else:
        tracks = np.empty((0, 10))
        max_obj_id = 0
    
    tracks_2 = np.loadtxt(os.path.join('./plugins/NeuralSORT', video, 'pedestrian', 'predict.txt'), delimiter=',')
    if len(tracks_2.shape) == 2:
        tracks_2[:, 1] += max_obj_id
        tracks = np.concatenate((tracks, tracks_2), axis=0)

    tracks = tracks[np.lexsort([tracks[:, 1], tracks[:, 0]])]  # frame -> ID
    lines_full = tracks

    vid = os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02", video)

    H, W = RESOLUTION[video]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs(f"./plugins/{exp}/NeuralSORT", exist_ok=True)
    out = cv2.VideoWriter(os.path.join(f"./plugins/{exp}/NeuralSORT", video + ".mp4"), fourcc, 10.0, (W, H))

    frames = sorted(os.listdir(vid))
    for frame in frames:
        img = cv2.imread(os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02", video, frame))
        # add bounding box for all tracks
        for line in lines_full:
            frame_id = int(line[0])
            if int(frame.split(".")[0]) == frame_id:
                bbox = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                start = int(bbox[0]), int(bbox[1])
                end = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                color = (0, 255, 0) # green
                thickness = 3
                cv2.rectangle(img, start, end, color, thickness)

        out.write(img)

def draw(video, seq, exp):
    path_gt = os.path.join(f"./plugins/{exp}/results", video, seq, "gt.txt")
    with open(path_gt, "r", encoding="utf-8") as file:
        lines_gt = file.readlines()

    path_predict = os.path.join(f"./plugins/{exp}/results", video, seq, "predict.txt")
    with open(path_predict, "r", encoding="utf-8") as file:
        lines_predict = file.readlines()

    cnt_gt = 0

    vid = os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02", video)

    H, W = RESOLUTION[video]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.join(f"./plugins/{exp}/visualize", video), exist_ok=True)
    out = cv2.VideoWriter(os.path.join(f"./plugins/{exp}/visualize", video, seq + ".mp4"), fourcc, 10.0, (W, H))
    
    frames = sorted(os.listdir(vid))
    for frame in frames:
        img = cv2.imread(os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02", video, frame))

        # add bounding box for ground-truth
        while True:
            if cnt_gt >= len(lines_gt):
                break
            line = lines_gt[cnt_gt].strip().split(" ")
            frame_id = int(line[0])
            if int(frame.split(".")[0]) < frame_id:
                break
            elif int(frame.split(".")[0]) > frame_id:
                cnt_gt = cnt_gt + 1
            else:
                bbox = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                start = int(bbox[0]), int(bbox[1])
                end = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                color = (255, 0, 0) # blue
                thickness = 3
                cv2.rectangle(img, start, end, color, thickness)
                cnt_gt = cnt_gt + 1
        
        # add bounding box for prediction
        for line_predict in lines_predict:
            line = line_predict.strip().split(",")
            frame_id = int(line[0].split(".")[0])
            if int(frame.split(".")[0]) == frame_id:
                bbox = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                start = int(bbox[0]), int(bbox[1])
                end = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                color = (0, 0, 255) # red
                thickness = 3
                cv2.rectangle(img, start, end, color, thickness)

        out.write(img)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="my_exp")
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--max_score", type=float, default=0.0)
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    path = f"./plugins/{args.exp}/results/pedestrian_detailed.csv"
    data = pd.read_csv(path, delimiter=',', header=0)
    data = data.sort_values(by="HOTA___AUC")

    motion_word = ["moving", "motion", "swift", "rapid", "quicker", "changing", "slowed", "stopping", "making", "came", "exceed", "used", "decelerating", "slowing", 
                   "rapid", "faster", "speedier", "transferring", "outpace", "applied", "move", "relocating", "turning", "driving", "coming", "standing", "walking", 
                   "movement", "transit", "parked", "parking", "parked,", "stationary", "going", "traveling", "turned"]
    color_word = ["white", "blue", "yellow", "green", "silver", "red", "color", "black", "light", "direction", "opposite"]
    pos_word = ["front", "positioned", "located", "situated", "ahead", "right", "left", "before", "camera", "upright"]
    
    cases = {}

    for idx, item in data.iterrows():
        dir = item["seq"].split("+")
        if len(dir) == 1:
            continue
        video, seq = dir[0], dir[1]
        if video not in cases.keys():
            cases[video] = []
        score = float(item["HOTA___AUC"])
        if args.min_score < score <= args.max_score:
            cases[video].append((seq))

    data = []
    for video in cases.keys():
        for seq in cases[video]:
            check = False
            seq2 = seq.split("-")
            for word in motion_word:
                if word in seq2:
                    check = True
                    break
            for word in color_word:
                if word in seq2:
                    check = True
                    break
            if not check:
                for word in pos_word:
                    if word in seq2:
                        print(video, expression_conversion(seq))
                        if args.draw:
                            draw(video, seq, exp=args.exp)
                        break