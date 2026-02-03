import warnings
import argparse
from ultralytics import YOLO

dcn_version = 3.0

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Train a YOLO model with a given name and loss type.')

# 添加 name 参数
parser.add_argument('--name', type=str, required=True, help='The name of the YOLO model.')
# 添加 model yaml 的位置 参数
parser.add_argument('--model', type=str, required=True, help='The name of the YOLO model yaml')

# 添加 loss 参数
parser.add_argument('--loss', type=str, default='EMASlideLoss', help='The type of loss function to use.')

parser.add_argument('--dataset', type=str, default='GC-DET-new', help='dataset NEU-DET-flip')

# 解析命令行参数
args = parser.parse_args()

# 使用 args.name 和 args.loss 替代硬编码的 name 和 loss 变量
name = args.name
loss = args.loss
model = args.model
dataset = args.dataset

if __name__ == '__main__':
    
    model = YOLO(f'./cfg/{model}.yaml')
    
    name = f'{name}-{loss}'
    
    model.train(data=f'./dataset/{dataset}/data.yaml',
                cache=False,
                imgsz=200,
                epochs=200,
                batch=64,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD',  # using SGD
                # patience=0,  # close earlystop
                resume=True,  # 断点续训,YOLO初始化时选择last.pt
                # amp=False,  # close amp
                # fraction=0.2,
                project=f'summer/runs/train/{dataset}',
                name=f'{name}', 
                )