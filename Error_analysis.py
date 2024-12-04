import os
import cv2
import numpy as np
from utils import *
import pandas as pd
import argparse

def draw_NeuralSORT(video):
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

    cnt_full = 0

    vid = os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02", video)

    H, W = RESOLUTION[video]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join("./plugins/my_exp3/visualize", video + ".mp4"), fourcc, 10.0, (W, H))
    
    frames = sorted(os.listdir(vid))
    for frame in frames:
        img = cv2.imread(os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02", video, frame))
        # add bounding box for all tracks
        while True:
            if cnt_full >= len(lines_full):
                break
            line = lines_full[cnt_full]
            frame_id = int(float(line[0]))
            if int(frame.split(".")[0]) < frame_id:
                break
            elif int(frame.split(".")[0]) > frame_id:
                cnt_full = cnt_full + 1
            else:
                obj_id = int(float(line[1]))
                bbox = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                start = int(bbox[0]), int(bbox[1])
                end = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                color = ((obj_id*13)%255, (obj_id*29)%255, obj_id)
                thickness = 3
                cv2.rectangle(img, start, end, color, thickness)
                cnt_full = cnt_full + 1

        out.write(img)

def draw(video, seq, exp):
    path_gt = os.path.join(f"./plugins/{exp}/results", video, seq, "gt.txt")
    with open(path_gt, "r", encoding="utf-8") as file:
        lines_gt = file.readlines()

    path_predict = os.path.join(f"./plugins/{exp}/results", video, seq, "predict.txt")
    with open(path_predict, "r", encoding="utf-8") as file:
        lines_predict = file.readlines()

    cnt_gt = 0
    cnt_predict = 0

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
        while True:
            if cnt_predict >= len(lines_predict):
                break
            line = lines_predict[cnt_predict].strip().split(",")
            frame_id = int(float(line[0]))
            if int(frame.split(".")[0]) < frame_id:
                break
            elif int(frame.split(".")[0]) > frame_id:
                cnt_predict = cnt_predict + 1
            else:
                bbox = float(line[2]), float(line[3]), float(line[4]), float(line[5])
                start = int(bbox[0]), int(bbox[1])
                end = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                color = (0, 0, 255) # red
                thickness = 3
                cv2.rectangle(img, start, end, color, thickness)
                cnt_predict = cnt_predict + 1

        out.write(img)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="my_exp")
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--max_score", type=float, default=0.0)
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    path = f"./plugins/{args.exp}/results/pedestrian_detailed.csv"
    print(path)
    data = pd.read_csv(path, delimiter=',', header=0)
    data = data.sort_values(by="HOTA___AUC")
    
    motion_word = ["moving", "motion", "swift", "rapid", "quicker", "changing", "slowed", "stopping", "making", "came", "exceed", "used", "decelerating", "slowing", 
                   "rapid", "faster", "speedier", "transferring", "outpace", "applied", "move", "relocating", "turning", "driving", "coming", "standing", "walking", 
                   "movement", "transit", "parked", "parking", "parked,", "stationary", "going", "traveling", "turned"]
    
    color_word = ["white", "blue", "yellow", "green", "silver", "red", "color", "black", "light", "direction"]
    
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
                        print(video, " ".join(seq2))
                        if args.draw:
                            draw(video, seq, exp=args.exp)
                        break