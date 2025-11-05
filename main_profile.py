import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('./summer/cfg//yolo11-TADDH-C3k2-DAttention.yaml')
    
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()
