import torch
from utils import *
from model25 import get_model_25, gen_sineembed_for_position
from dataloader import get_dataloader
from opts import opt
from os.path import join

if __name__ == "__main__":
    opt.model = 25
    opt.num_layers = 1
    opt.test_bs = 1
    opt.test_ckpt = "my_exp58/epoch149.pth"
    opt.sample_frame_len = 2
    opt.sample_frame_num = 1
    opt.sample_frame_stride = 2

    get_model = eval("get_model_" + str(opt.model))
    model = get_model(opt, 'Model')
    model, _ = load_from_ckpt(model, join(opt.save_root, f'{opt.test_ckpt}'))
    model.eval()

    dataloader = get_dataloader('test', opt, 'Track_Dataset')

    with torch.no_grad():
        for data in dataloader:
            input = dict(
                bbox = data['bbox'].cuda(),
                local_img=data['cropped_images'].cuda(),
                exp=tokenize(data['expression_new']).cuda(),
            )
            print(data['expression_new'])
            bbox = gen_sineembed_for_position(input['bbox'])
            output = model(input)
            # print(output["logits"])
            # break