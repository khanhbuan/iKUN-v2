import json
import os
from utils import *
import clip
import torch.nn.functional as F

data = {}

modes = ["train", "test"]

model, _ = clip.load('./plugins/CLIP/ViT-B-32.pt')
model.cuda()
model.eval()

for mode in modes:
    data[mode] = {}
    sum = 0
    for video in VIDEOS[mode]:
        path = os.path.join("./plugins/Refer-KITTI/expression", video)
        exps = os.listdir(path)
        for exp in exps:
            with open(os.path.join(path, exp), "r", encoding='utf-8') as data_file:
                label = json.load(data_file)
                sentence = label["sentence"]
                expression = expression_conversion(sentence)
                if expression not in data[mode].keys():
                    data[mode][expression] = {}
                
                if mode == 'train':
                    if 'probability' not in data[mode][expression].keys():
                        data[mode][expression]['probability'] = 0
                    data[mode][expression]['probability'] += 1
                    sum += 1

                if 'feature' not in data[mode][expression].keys():
                    data[mode][expression]['feature'] = []
                    text = clip.tokenize(exp).cuda()
                    feat = model.encode_text(text)
                    feat = F.normalize(feat, p=2)
                    feat = feat.detach().cpu().tolist()[0]
                    data[mode][expression]['feature'] = feat
    
    if mode == "train":
        for expression in data[mode].keys():
            num = data[mode][expression]['probability']
            data[mode][expression]['probability']  = (num * 1.0) / sum

with open('./plugins/textual_features.json', "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("successful!")