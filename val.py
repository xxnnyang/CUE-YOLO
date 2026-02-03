import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch

if __name__ == '__main__':

    name = 'yolov11-loss'
    
    model = YOLO(f'./runs/train/NEU-DET-flip/{name}/weights/best.pt')
    model = torch.compile(model)
    
    model.val(data='./dataset/NEU-DET-flip/data.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              plots=True,
              save_json=True, # if you need to cal coco metrice
              project='./summer/runs/val',
              name=f'NEU-DET-flip-{name}',
              )