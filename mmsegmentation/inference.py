import os
import mmcv
import torch
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import json
import numpy as np

def main(args):
    work_dir = args.work_dir
    cfg = Config.fromfile(args.model_config_path)
    checkpoint_path = args.model_ckpt_path
    root = args.test_imgfile_path
    output_filename = args.output_file_name

    # dataset config 수정
    cfg.data.test.img_dir = root
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 1
    cfg.work_dir = work_dir
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)


    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader)

    # submission 양식에 맞게 output 후처리
    input_size = 512
    output_size = 256

    submission = pd.read_csv("/opt/ml/input/code/submission/sample_submission.csv", index_col=None)
    json_dir = os.path.join("/opt/ml/input/data/test.json")

    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    # PredictionString
    for img_id, pred in enumerate(output):

        img_id = datas["images"][img_id]
        file_name = img_id["file_name"]

        temp_mask = []
        pred = pred.reshape(1, 512, 512)
        mask = pred.reshape((1, output_size, input_size//output_size, output_size, input_size//output_size)).max(4).max(2) # resize to 256*256
        temp_mask.append(mask)
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)
        string = oms.flatten()
        submission = pd.concat([submission, pd.DataFrame([{"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}])]
                                   ,ignore_index=True)
    submission.to_csv(os.path.join(cfg.work_dir, f'{output_filename}.csv'), index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--model_config_path', type=str)
    parser.add_argument('--model_ckpt_path', type=str)
    parser.add_argument('--test_imgfile_path', type=str)
    parser.add_argument('--output_file_name', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

# model -> dense crf -> inference tools
# model -> dense crf -> ensemble tools