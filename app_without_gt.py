import gradio as gr
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import time
from threading import Thread, Event
import queue

# ===== 路径配置 =====
IMG_DIR = Path("./GC10/images/")
GT_TXT_DIR = Path("./GC10/txt")
WEIGHTS_YOLOV11 = "./yolov11.pt"
# WEIGHTS_DEFECT = "./defect_yolov11.pt"
WEIGHTS_DEFECT = "./best.pt"

# ===== 类别名设置 =====
GT_CLASS_LABELS = [
    'Welding Line', 'Punching Hole', 'Silk Spot', 'Crease', 'Waist Folding',
    'Water Spot', 'Crease gap', 'Inclusion', 'Oil Spot', 'Rolled Pit', 'Waist Folding'
]

PRED_CLASS_LABELS = [
    'Punching Hole', 'Welding Line', 'Crease Gap', 'Water Spot', 'Oil Spot', 
    'Silk Spot', 'Inclusion', 'Rolled Pit', 'Crease', '10', '11'
]

# ===== 类别颜色映射 =====
CLASS_COLOR_MAP = {
    'Welding Line': (255, 50, 50),
    'Punching Hole': (50, 255, 50),
    'Silk Spot': (50, 150, 255),
    'Crease': (255, 255, 50),
    'Waist Folding': (255, 50, 255),
    'Water Spot': (50, 255, 255),
    'Crease gap': (255, 180, 50),
    'Crease Gap': (255, 180, 50),
    'Inclusion': (200, 50, 255),
    'Oil Spot': (50, 255, 150),
    'Rolled Pit': (255, 220, 50),
    '5': (255, 150, 150),
    '6': (150, 255, 150),
    '7': (150, 200, 255),
    '10': (255, 255, 150),
    '11': (255, 150, 255),
}

DEFAULT_COLOR = (200, 200, 200)

def get_color_for_class(class_name):
    """根据类别名称获取固定颜色"""
    return CLASS_COLOR_MAP.get(class_name, DEFAULT_COLOR)

# ===== 绘图参数 =====
FONT_SCALE = 1.0
FONT_THICKNESS = 2
BOX_THICKNESS = 2

def draw_boxes(img_bgr, boxes_xyxy, cls_ids, labels, confidences=None):
    """绘制检测框"""
    img = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        cid = int(cls_ids[i])
        
        if 0 <= cid < len(labels):
            label_text = f"{labels[cid]}"
            color = get_color_for_class(labels[cid])
        else:
            label_text = f"Class {cid}"
            color = DEFAULT_COLOR

        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        if confidences is not None and i < len(confidences):
            conf = confidences[i]
            label_text = f"{label_text} {conf:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(label_text, font, FONT_SCALE, FONT_THICKNESS)
        
        y_text_top = y1 - text_h - 10
        if y_text_top < 0:
            y_text_top = 0
        
        padding = 5
        cv2.rectangle(img, 
                     (x1, y_text_top), 
                     (x1 + text_w + padding, y_text_top + text_h + 10), 
                     color, -1)
        
        cv2.putText(img, label_text, 
                   (x1 + padding // 2, y_text_top + text_h + 2),
                   font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
    
    return img

def read_gt_yolo_to_xyxy(txt_path: Path, W: int, H: int):
    """读取YOLO格式标注"""
    if not txt_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int32)

    xyxy_list, cls_list = [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            cx *= W; cy *= H; w *= W; h *= H
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            xyxy_list.append([x1, y1, x2, y2])
            cls_list.append(cid)
    if not xyxy_list:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int32)
    return np.array(xyxy_list, dtype=np.float32), np.array(cls_list, dtype=np.int32)

# 加载模型
print("Loading models...")
model_yolov11 = None
model_defect = None

try:
    if Path(WEIGHTS_YOLOV11).exists():
        model_yolov11 = YOLO(WEIGHTS_YOLOV11)
        print("✓ YOLOv11 model loaded")
except Exception as e:
    print(f"✗ Error loading YOLOv11: {e}")

try:
    if Path(WEIGHTS_DEFECT).exists():
        model_defect = YOLO(WEIGHTS_DEFECT)
        print("✓ CUE-YOLOv11 model loaded")
except Exception as e:
    print(f"✗ Error loading CUE-YOLOv11: {e}")

def get_available_images():
    """获取可用的图像列表"""
    if not IMG_DIR.exists():
        return []
    images = sorted([f.stem for f in IMG_DIR.glob("*.jpg")])
    return images

def create_comparison_image(img, yolo_results, defect_results, 
                           image_name, conf_threshold, iou_threshold):
    """创建两模型对比图"""
    H, W = img.shape[:2]
    
    # YOLOv11结果
    if yolo_results is not None and len(yolo_results) > 0:
        yolo_vis = draw_boxes(img, yolo_results[0], yolo_results[1], 
                             PRED_CLASS_LABELS, yolo_results[2])
    else:
        yolo_vis = img.copy()
    
    # CUE-YOLOv11结果
    if defect_results is not None and len(defect_results) > 0:
        defect_vis = draw_boxes(img, defect_results[0], defect_results[1], 
                               PRED_CLASS_LABELS, defect_results[2])
    else:
        defect_vis = img.copy()
    
    # 添加标题
    title_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 创建标题栏
    def add_title(img, title, color=(255, 255, 255)):
        titled = np.zeros((title_height, W, 3), dtype=np.uint8)
        titled[:] = (40, 40, 40)  # 深灰色背景
        
        text_size = cv2.getTextSize(title, font, 1.2, 2)[0]
        text_x = (W - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        
        cv2.putText(titled, title, (text_x, text_y), 
                   font, 1.2, color, 2, cv2.LINE_AA)
        
        return np.vstack([titled, img])
    
    yolo_vis = add_title(yolo_vis, "YOLOv11", (100, 200, 255))
    defect_vis = add_title(defect_vis, "CUE-YOLOv11", (255, 200, 100))
    
    # 水平拼接
    combined = np.hstack([yolo_vis, defect_vis])
    
    # 添加顶部信息栏
    info_height = 80
    info_bar = np.zeros((info_height, combined.shape[1], 3), dtype=np.uint8)
    info_bar[:] = (30, 30, 30)
    
    # 图像名称
    cv2.putText(info_bar, f"Image: {image_name}", (20, 30), 
               font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 参数信息
    info_text = f"Conf: {conf_threshold:.2f} | IoU: {iou_threshold:.2f}"
    cv2.putText(info_bar, info_text, (20, 60), 
               font, 0.8, (200, 200, 200), 1, cv2.LINE_AA)
    
    # 检测数量
    yolo_count = len(yolo_results[0]) if yolo_results is not None and len(yolo_results) > 0 else 0
    defect_count = len(defect_results[0]) if defect_results is not None and len(defect_results) > 0 else 0
    
    count_text = f"YOLOv11: {yolo_count} | CUE-YOLOv11: {defect_count}"
    text_size = cv2.getTextSize(count_text, font, 0.8, 1)[0]
    cv2.putText(info_bar, count_text, 
               (combined.shape[1] - text_size[0] - 20, 45), 
               font, 0.8, (100, 255, 255), 1, cv2.LINE_AA)
    
    # 合并信息栏和图像
    final = np.vstack([info_bar, combined])
    
    return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

class SlideShow:
    """幻灯片播放控制器"""
    def __init__(self):
        self.is_playing = False
        self.stop_event = Event()
        self.current_index = 0
        self.images = []
        
    def start(self, images, interval, conf, iou, show_yolo, show_defect):
        """开始播放"""
        self.images = images
        self.interval = interval
        self.conf = conf
        self.iou = iou
        self.show_yolo = show_yolo
        self.show_defect = show_defect
        self.is_playing = True
        self.stop_event.clear()
        self.current_index = 0
        
    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.stop_event.set()
        
    def get_current_frame(self):
        """获取当前帧"""
        if not self.images or self.current_index >= len(self.images):
            return None, "播放结束"
            
        image_name = self.images[self.current_index]
        img_path = IMG_DIR / f"{image_name}.jpg"
        
        img = cv2.imread(str(img_path))
        if img is None:
            return None, f"无法加载: {image_name}"
        
        H, W = img.shape[:2]
        
        # YOLOv11
        yolo_results = None
        if self.show_yolo and model_yolov11 is not None:
            r1 = model_yolov11.predict(source=img, conf=self.conf, iou=self.iou, verbose=False)[0]
            if r1.boxes is not None:
                xyxy1 = r1.boxes.xyxy.cpu().numpy()
                cls1 = r1.boxes.cls.cpu().numpy().astype(int)
                conf1 = r1.boxes.conf.cpu().numpy()
                yolo_results = (xyxy1, cls1, conf1)
        
        # CUE-YOLOv11
        defect_results = None
        if self.show_defect and model_defect is not None:
            r2 = model_defect.predict(source=img, conf=self.conf, iou=self.iou, verbose=False)[0]
            if r2.boxes is not None:
                xyxy2 = r2.boxes.xyxy.cpu().numpy()
                cls2 = r2.boxes.cls.cpu().numpy().astype(int)
                conf2 = r2.boxes.conf.cpu().numpy()
                defect_results = (xyxy2, cls2, conf2)
        
        # 创建对比图
        result_img = create_comparison_image(
            img, yolo_results, defect_results,
            image_name, self.conf, self.iou
        )
        
        info = f"播放中: {self.current_index + 1}/{len(self.images)} - {image_name}"
        
        return result_img, info

slideshow = SlideShow()

def play_slideshow(interval, conf, iou, show_yolo, show_defect, loop):
    """播放幻灯片"""
    images = get_available_images()
    
    if not images:
        yield None, "没有找到图像"
        return
    
    slideshow.start(images, interval, conf, iou, show_yolo, show_defect)
    
    while slideshow.is_playing:
        # 获取当前帧
        frame, info = slideshow.get_current_frame()
        yield frame, info
        
        # 等待
        for _ in range(int(interval * 10)):
            if not slideshow.is_playing:
                break
            time.sleep(0.1)
        
        # 下一张
        slideshow.current_index += 1
        
        # 检查是否结束
        if slideshow.current_index >= len(images):
            if loop:
                slideshow.current_index = 0  # 循环
                yield frame, "循环播放..."
            else:
                yield frame, "播放完成"
                slideshow.stop()
                break

def stop_slideshow():
    """停止播放"""
    slideshow.stop()
    return None, "已停止"

# 创建Gradio界面
with gr.Blocks(title="检测效果循环播放", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 缺陷检测效果循环播放")
    gr.Markdown("自动循环播放所有图像的检测效果对比")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 播放设置")
            
            interval_slider = gr.Slider(
                minimum=0.5,
                maximum=10.0,
                value=2.0,
                step=0.5,
                label="播放间隔 (秒)",
                info="每张图片的显示时间"
            )
            
            loop_checkbox = gr.Checkbox(
                label="循环播放",
                value=True,
                info="播放完毕后重新开始"
            )
            
            gr.Markdown("### 🎯 检测参数")
            
            conf_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="置信度阈值"
            )
            
            iou_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.45,
                step=0.05,
                label="NMS IoU阈值"
            )
            
            gr.Markdown("### 👁️ 显示选项")
            
            show_yolov11 = gr.Checkbox(label="显示YOLOv11", value=True)
            show_defect = gr.Checkbox(label="显示CUE-YOLOv11", value=True)
            
            gr.Markdown("### 🎮 控制")
            
            with gr.Row():
                play_btn = gr.Button("▶️ 开始播放", variant="primary", size="lg")
                stop_btn = gr.Button("⏸️ 停止", variant="stop", size="lg")
            
            status_box = gr.Textbox(
                label="状态信息",
                lines=5,
                interactive=False
            )
            
            # 图像统计
            images = get_available_images()
            gr.Markdown(f"**📊 共有 {len(images)} 张图像**")
    
        with gr.Column(scale=3):
            output_image = gr.Image(
                label="检测效果对比",
                type="numpy",
                height=800
            )
    
    # 绑定事件
    play_event = play_btn.click(
        fn=play_slideshow,
        inputs=[interval_slider, conf_slider, iou_slider, show_yolov11, show_defect, loop_checkbox],
        outputs=[output_image, status_box]
    )
    
    stop_btn.click(
        fn=stop_slideshow,
        outputs=[output_image, status_box],
        cancels=[play_event]
    )
    
    gr.Markdown("""
    ### 📖 使用说明
    
    1. **设置播放间隔**: 调整每张图片的显示时间（0.5-10秒）
    2. **选择循环模式**: 勾选"循环播放"可以无限循环
    3. **调整检测参数**: 设置置信度和IoU阈值
    4. **选择显示内容**: 可以选择显示哪些检测结果
    5. **开始播放**: 点击"开始播放"按钮
    6. **停止播放**: 随时点击"停止"按钮暂停
    
    **💡 提示**:
    - 图像按两列显示: YOLOv11 | CUE-YOLOv11
    - 顶部显示当前图像名称和检测参数
    - 可以在播放过程中调整参数（需要重新开始播放）
    """)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7861,  # 使用不同的端口
        share=False,
        inbrowser=True
    )