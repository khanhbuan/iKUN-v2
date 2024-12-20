# import `opts` first to set gpus
from opts import opt

from utils import set_seed
set_seed(opt.seed)

import wandb
import time
import shutil
import math
from os.path import join, exists

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from loss import *
from utils import *
from test import test_accuracy
from dataloader import get_dataloader

from model0 import get_model_0
from model1 import get_model_1
from model2 import get_model_2
from model3 import get_model_3
from model4 import get_model_4
from model5 import get_model_5
from model6 import get_model_6
from model7 import get_model_7
from model8 import get_model_8
from model9 import get_model_9
from model10 import get_model_10
from model11 import get_model_11
from model12 import get_model_12
from model13 import get_model_13
from model14 import get_model_14
from model15 import get_model_15
from model16 import get_model_16
from model17 import get_model_17
from model18 import get_model_18
from model19 import get_model_19
from model20 import get_model_20
from model21 import get_model_21
from model22 import get_model_22
from model23 import get_model_23
from model24 import get_model_24
from model25 import get_model_25

scaler = GradScaler()

get_model = eval("get_model_" + str(opt.model))

model = get_model(opt, 'Model')

sim_loss = SimilarityLoss(
    rho=opt.loss_rho,
    gamma=opt.loss_gamma,
    reduction=opt.loss_reduction,
)

optimizer = optim.AdamW(
    [{'params': model.parameters()},],
    lr=opt.base_lr,
    weight_decay=opt.weight_decay,
)

if opt.resume_path:
    model, resume_epoch = load_from_ckpt(model,  opt.resume_path)
else:
    resume_epoch = -1
    if exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)

save_configs(opt)
logger = get_logger(opt.save_dir)
writer = SummaryWriter(opt.save_dir)

dataloader_train = get_dataloader('train', opt, 'RMOT_Dataset', show=True)
dataloader_test = get_dataloader('test', opt, 'RMOT_Dataset', show=False)

print(
    '========== Training (Text-Guided {}) =========='
        .format('ON' if opt.kum_mode else 'OFF')
)
iteration = 0
logger.info('Start training!')

wandb.login(key = '65f5d63dba6bb11e5e8b27ee37519ce1168b1729')
wandb.init(project="iKUN")

for epoch in range(resume_epoch + 1, opt.max_epoch):
    model.train()
    BATCH_TIME = AverageMeter('Time', ':6.3f')
    LOSS = AverageMeter('Loss', ':.4e')
    lr = get_lr(opt, epoch)
    set_lr(optimizer, lr)
    meters = [BATCH_TIME, LOSS]
    PROGRESS = ProgressMeter(
        num_batches=len(dataloader_train),
        meters=meters,
        prefix="Epoch [{}/{}] ".format(epoch, opt.max_epoch),
        lr=lr
    )
    end = time.time()
    # train
    for batch_idx, data in enumerate(dataloader_train):
        # load
        expression = data['target_expressions']
        # forward
        inputs = dict(
            local_img=data['cropped_images'].cuda(),
            bbox = data['bbox'].cuda(),
            exp=tokenize(expression).cuda(),
        )
        targets = data['target_labels'].view(-1).cuda()
        logits = model(inputs, epoch)['logits']
        # loss
        loss = sim_loss(logits, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # write
        BATCH_TIME.update(time.time() - end)
        LOSS.update(loss.item(), opt.train_bs)
        end = time.time()
        iteration += 1
        writer.add_scalar('Train/LR', lr, iteration)
        writer.add_scalar('Loss/', loss.item(), iteration)
        if (batch_idx + 1) % opt.train_print_freq == 0:
            PROGRESS.display(batch_idx)
            logger.info(
                'Epoch:[{}/{}] [{}/{}] Loss:{:.5f}'
                    .format(epoch, opt.max_epoch, batch_idx, len(dataloader_train), loss.item())
            )

    # test
    torch.cuda.empty_cache()
    if (epoch + 1) % opt.eval_frequency == 0:
        p, r = test_accuracy(model, dataloader_test)
        log_info = 'precision: {:.2f}% / recall: {:.2f}%'.format(p, r)
        logger.info(log_info)
        print(log_info)
        p, r = p.item(), r.item()
        if math.isnan(p):
            p = 0
        if math.isnan(r):
            r = 0
        wandb.log({"loss": loss.item(), "precision": p, "recall": r})
    if (epoch + 1) % opt.save_frequency == 0:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer,
            'epoch': epoch,
        }
        torch.save(state_dict, join(opt.save_dir, f'epoch{epoch}.pth'))
    torch.cuda.empty_cache()

wandb.finish()
logger.info('Finish training!')