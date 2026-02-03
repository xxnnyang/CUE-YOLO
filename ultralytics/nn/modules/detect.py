# 导入必要的库
from ultralytics import YOLO


# ---------- 加载模型 ----------
# 方法 1：通过 yaml 文件新建一个模型 (根据 yaml 文件中的模型定义自动搭建一个模型)
model = YOLO('./ultralytics/cfg/models/summer/yolo11-TADDH-C3k2-DAttention.yaml')
