from importlib import import_module
from pathlib import Path
import os
import random
import time
import json
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloaders.DataLoader import CustomDataLoader
from loss import create_criterion
from scheduler import create_scheduler

import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import MetricCollection

import albumentations as A
from albumentations.pytorch import ToTensorV2
from optimizer import AdamP

import wandb
import yaml
from easydict import EasyDict

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from utils.utils import rand_bbox, copyblob
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_model(model, saved_dir, file_name='best_model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def collate_fn(batch):
    return tuple(zip(*batch))

def save_table(table_name, model, val_loader, device):
    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types = True)

    for step, (im, mask, _) in tqdm(enumerate(val_loader), total = len(val_loader)):

        im = torch.stack(im)       
        mask = torch.stack(mask).long()

        im, mask, = im.to(device), mask.to(device)

        _mask = model(im)
        _, _mask = torch.max(_mask, dim=1)

        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(im[0].permute(1,2,0).detach().cpu()[:,:,0])
        plt.savefig("original_image.jpg")
        plt.close()

        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(mask.permute(1,2,0).detach().cpu()[:,:,0])
        plt.savefig("original_mask.jpg")
        plt.close()

        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(_mask.permute(1,2,0).detach().cpu()[:,:,0])
        plt.savefig("predicted_mask.jpg")
        plt.close()

        table.add_data(
            wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
        )

    wandb.log({table_name: table})

def save_table(table_name, model, val_loader, device):
  table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types = True)

  for step, (im, mask, _) in tqdm(enumerate(val_loader), total = len(val_loader)):

    im = torch.stack(im)       
    mask = torch.stack(mask).long()

    im, mask, = im.to(device), mask.to(device)

    _mask = model(im)
    _, _mask = torch.max(_mask, dim=1)

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(im[0].permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("original_image.jpg")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(mask.permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("original_mask.jpg")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(_mask.permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("predicted_mask.jpg")
    plt.close()

    table.add_data(
        wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
        wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
        wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
    )


  wandb.log({table_name: table})

def train(args):
    print(f'Start training...')

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentations
    train_transform = A.Compose([
                            # A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=384, width=384, p=0.5),
                            A.Resize(512, 512),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                            # A.Resize(256,256),
                            ToTensorV2()
                            ])

    val_transform = A.Compose([
                            ToTensorV2()
                          ])
    # -- data_set
    train_path = dataset_path + args.train_path
    val_path = dataset_path + args.val_path

    train_dataset = CustomDataLoader(data_dir=train_path, dataset_path=dataset_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, dataset_path=dataset_path, mode='val', transform=val_transform)

    # -- datalodaer
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=args.valid_batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn,
                            drop_last=True)
                                         
    # -- model
    model_module = getattr(smp, args.decoder)
    model = model_module(
        encoder_name=args.encoder, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                     # model output channels (number of classes in your dataset)
    )
    # device 할당
    model = model.to(device)   

    # -- loss & metric
    criterion = create_criterion(args.criterion)

    # -- optimizer
    if args.optimizer == "AdamP" :
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    else :
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6
        )

    # # -- scheduler
    if args.scheduler:
        scheduler = create_scheduler(optimizer, args.scheduler, args.epochs, args.lr)

    # -- train
    n_class = 11
    best_loss, best_mIoU = np.inf, 0
    val_every = 1
    
    # Grad accumulation
    NUM_ACCUM = args.grad_accum
    optimizer.zero_grad()
    
    # Early Stopping
    PATIENCE = args.patience
    counter = 0

    # average = macro가 기본 옵션입니다
    hist = MulticlassJaccardIndex(num_classes=n_class).cuda()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    coco = train_dataset.get_coco()
    for epoch in range(args.epochs):
        model.train()

        with tqdm(total=len(train_loader)) as pbar:
            for step, (images, masks, _) in enumerate(train_loader):
                images = torch.stack(images)
                masks = torch.stack(masks)

                if args.copyblob:
                    for i in range(images.size()[0]):
                        # Paper Pack
                        copyblob(coco, transform=train_transform, dst_img=images[i], dst_mask=masks[i], src_class=3, dst_class=0, p=0.1)
                        # Metal 
                        copyblob(coco, transform=train_transform, dst_img=images[i], dst_mask=masks[i], src_class=4, dst_class=0, p=0.1)
                        # Glass
                        copyblob(coco, transform=train_transform, dst_img=images[i], dst_mask=masks[i], src_class=5, dst_class=0, p=0.2) 
                        # Battery
                        copyblob(coco, transform=train_transform, dst_img=images[i], dst_mask=masks[i], src_class=9, dst_class=0, p=0.2)
                        # Cloth
                        copyblob(coco, transform=train_transform, dst_img=images[i], dst_mask=masks[i], src_class=10, dst_class=0, p=0.3) 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.long().to(device)
                
                # generate mixed sample
                if args.cutmix:
                    lam = np.random.beta(1., 1.)
                    rand_index = torch.randperm(images.size()[0]).cuda()
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    masks[:, bbx1:bbx2, bby1:bby2] = masks[rand_index, bbx1:bbx2, bby1:bby2]

                # inference
                outputs = model(images)
                images, masks = images.to(device), masks.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp) :
                            
                    # inference
                    outputs = model(images)
                
                    # loss 계산 (cross entropy loss)
                    loss = criterion(outputs, masks)

                # loss.backward()
                scaler.scale(loss).backward()

                if step % NUM_ACCUM == 0:
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # batch에 대한 mIoU 계산, baseline code는 누적을 계산합니다
                mIoU = hist(outputs, masks).item()
                pbar.update(1)
                
                logging = {
                    'Tr Loss': round(loss.item(),4),
                    'Tr mIoU': round(mIoU,4),
                }
                pbar.set_postfix(logging)

                # step 주기에 따른 loss 출력
                if (step + 1) % args.log_interval == 0:
                    current_lr = get_lr(optimizer)
                    logging['lr'] = current_lr
                    # wandb
                    wandb.log(logging)
            
        hist.reset()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU, IoU_by_class = validation(model, val_loader, device, criterion, epoch, args)
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = val_mIoU
                save_model(model, saved_dir)
                counter = 0
            else:
                counter += 1

            # wandb
            wandb.log(
                {'Val Loss': avrg_loss, 'Val mIoU': val_mIoU}
            )
            wandb.log(IoU_by_class)

            if counter > PATIENCE:
                print('Early Stopping...')
                break
        
        if args.scheduler:
            scheduler.step()  

    save_table("Predictions", model, val_loader, device)

def validation(model, data_loader, device, criterion, epoch, args):
    print(f'Start validation!')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        # metric의 묶음을 한 번에 사용
        metric_collection = MetricCollection({
            "micro": MulticlassJaccardIndex(num_classes=n_class, average="micro"),
            "macro": MulticlassJaccardIndex(num_classes=n_class, average="macro"),      # mIoU
            "classwise": MulticlassJaccardIndex(num_classes=n_class, average="none")    # classwise IoU
        })
        metric_collection.cuda()

        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            metric_collection.update(outputs, masks)
        
        result = metric_collection.compute()
        micro = result["micro"].item()
        macro = result["macro"].item()
        classwise_results = result["classwise"].detach().cpu().numpy()
        category_list = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
                'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        IoU_by_class = {classes : round(IoU, 4) for IoU, classes in zip(classwise_results, category_list)}

        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch+1} || Average Loss: {round(avrg_loss.item(), 4)} || macro : {round(macro, 4)} || micro: {round(micro, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss, macro, IoU_by_class

if __name__ == "__main__":

    dataset_path = '/opt/ml/input/data'
    CONFIG_FILE_NAME = "./config/config.yaml"
    with open(CONFIG_FILE_NAME, "r") as yml_config_file:
        args = yaml.load(yml_config_file, Loader=yaml.FullLoader)
        args = EasyDict(args["train"])

    print(args)
    seed_everything(args.seed)

    saved_dir = './saved/' + args.experiment_name
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    CFG = {
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
        "seed" : args.seed,
        "encoder" : args.encoder,
        "encoder_weights" : args.encoder_weights,
        "decoder" : args.decoder,
        "optimizer" : args.optimizer,
        "scheduler" : args.scheduler,
        "criterion" : args.criterion,
    }

    wandb.init(
        project=args.project, entity=args.entity, name=args.experiment_name, config=CFG,
    )

    wandb.define_metric("Tr Loss", summary="min")
    wandb.define_metric("Tr mIoU", summary="max")

    wandb.define_metric("Val Loss", summary="min")
    wandb.define_metric("Val mIoU", summary="max")

    train(args)

    wandb.finish()

