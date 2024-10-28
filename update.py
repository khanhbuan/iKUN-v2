import os
import cv2
import json
import torch
from torchvision.transforms import Resize
from TransReID.model import load_config, make_model

if __name__ == "__main__":
    cfg1 = load_config("./TransReID/Market/vit_transreid_stride.yml")
    model1 = make_model(cfg1, num_class=751, camera_num=6, view_num=0)
    model1.load_param("./TransReID/Market/vit_transreid_market.pth")
    model1.eval()
    model1.to("cuda:1")
    transform_person = Resize((256, 128)).to("cuda:1")

    # cfg2 = load_config("./TransReID/VehicleID/vit_transreid_stride.yml")
    # model2 = make_model(cfg2, num_class=13164, camera_num=0, view_num=2)
    # model2.load_param("./TransReID/VehicleID/vit_transreid_vehicleID.pth")
    # model2.eval()
    # model2.to("cuda:2")
    # transform_car = Resize((256, 256)).to("cuda:2")

    path = "./plugins/Refer-KITTI_labels.json"
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for video in data.keys():
        for obj_id in data[video].keys():
            for frame_id in data[video][obj_id].keys():
                frame_dict = data[video][obj_id][frame_id]

                category = frame_dict['category'][0]
                if category == "car":
                    continue
                bbox = frame_dict['bbox']
                img_path = os.path.join("./plugins/Refer-KITTI/KITTI/training/image_02",
                                        video,
                                        frame_id.zfill(6)+".png"
                                        )
                img = cv2.imread(img_path)
                H, W, _ = img.shape
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                x_min, y_min, x_max, y_max = int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)

                img = torch.from_numpy(img[y_min:y_max, x_min:x_max]).permute(2, 0, 1).to("cuda:1")
                img = transform_person(img)[None,:,:,:].to(torch.float32)
                embed = model1(img, cam_label=0).to("cpu")
                data[video][obj_id][frame_id]["embed"] = embed
                """
                else:
                    img = torch.from_numpy(img[y_min:y_max, x_min:x_max]).permute(2, 0, 1).to("cuda:2")
                    img = transform_car(img)[None,:,:,:].to(torch.float32)
                    embed = model2(img, view_label=0)
                """
    

    with open('./plugins/Refer-KITTI_labels-v3.json', "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("finish!")