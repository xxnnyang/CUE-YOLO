import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

# YOLO11n summary (fused): 238 layers, 2,583,907 parameters, 0 gradients, 6.3 GFLOPs
# model weights:./runs/runs/train/GC-DET-yolo11/weights/best.pt size:5.2M (bs:32)Latency:0.00094s +- 0.00136s fps:1068.8
# python get_FPS.py --weights ./runs/runs/train/GC-DET-yolo11/weights/best.pt  --batch 32

# YOLO11-C3k2-DAttention summary (fused): 266 layers, 2,618,915 parameters, 0 gradients, 6.4 GFLOPs
# model weights:./runs/runs/train/GC-DET-yolo11-C3k2-DAttention/weights/best.pt size:5.3M (bs:32)Latency:0.00113s +- 0.00172s fps:885.7
# python get_FPS.py --weights ./runs/runs/train/GC-DET-yolo11-C3k2-DAttention/weights/best.pt  --batch 32

# YOLO11-TADDH summary (fused): 218 layers, 2,200,724 parameters, 0 gradients, 7.9 GFLOPs
# model weights:./runs/runs/train/GC-DET-yolo11-TADDH/weights/best.pt size:8.7M (bs:32)Latency:0.00124s +- 0.00157s fps:803.4
# python get_FPS.py --weights ./runs/runs/train/GC-DET-yolo11-TADDH/weights/best.pt  --batch 32

# YOLO11-TADDH-C3k2-DAttention summary (fused): 246 layers, 2,235,732 parameters, 0 gradients, 7.9 GFLOPs
# model weights:./runs/runs/train/GC-DET-yolo11-TADDH-C3k2-DAttention/weights/best.pt size:4.5M (bs:32)Latency:0.00141s +- 0.00182s fps:708.1
# python get_FPS.py --weights ./runs/runs/train/GC-DET-yolo11-TADDH-C3k2-DAttention/weights/best.pt  --batch 32

# python get_FPS.py --weights ./runs/runs/train/GC-DET-yolo11-TADDH-C3k2-DAttention/weights/best.pt  --batch 32
# python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolo11-C3k2-Star-CAA-EMASlideLoss/weights/best.pt --batch 1
# python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolo11-C3k2-Star-CAA-TADDH-loss2/weights/best.pt --batch 1
# python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolov11-loss/weights/best.pt --batch 1

# python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolo11-C3k2-Star-EMASlideLoss/weights/best.pt --batch 1
# python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolo11-starnet-EMASlideLoss/weights/best.pt --batch 1

#  python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolo11-starnet-EMASlideLoss/weights/best.pt --batch 1
#  python get_FPS.py --weights ./summer/runs/train/NEU-DET-flip/yolov11-loss/weights/best.pt --batch 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=200, type=int, help='warmup time')
    parser.add_argument('--testtime', default=1000, type=int, help='test time')
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    opt = parser.parse_args()
    
    device = select_device(opt.device, batch=opt.batch)
    
    # Model
    weights = opt.weights
    if weights.endswith('.pt'):
        model = attempt_load_weights(weights, device=device, fuse=True)
        print(f'Loaded {weights}')  # report
    else:
        model = YOLO(weights).model
        model.fuse()
        
    model = model.to(device)
    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)
    
    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()
    
    print('begin warmup...')
    for i in tqdm(range(opt.warmup), desc='warmup....'):
        model(example_inputs)
    
    print('begin test latency...')
    time_arr = []
    
    for i in tqdm(range(opt.testtime), desc='test latency....'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)
    
    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)
    
    if weights.endswith('.pt'):
        print(f'model weights:{opt.weights} size:{get_weight_size(opt.weights)}M (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}')
    else:
        print(f'model yaml:{opt.weights} (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}')