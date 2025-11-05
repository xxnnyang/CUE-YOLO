import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    name = 'yolov11-loss'
    
    model = YOLO(f'./summer/runs/train/NEU-DET-flip/{name}/weights/best.pt')
    model.predict(
                  source='./dataset/NEU-DET/images/train',
                  imgsz=640,
                  # project='runs/detect',
                  project='./summer/runs/detect',
                  name=name,
                  save=True,
                  conf=0.5,
                )