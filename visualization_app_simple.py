import gradio as gr
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ===== 路径配置 =====
IMG_DIR = Path("./GC10/images/")
GT_TXT_DIR = Path("./GC10/txt")
WEIGHTS_YOLOV11 = "./yolov11.pt"
# WEIGHTS_DEFECT = "./defect_yolov11.pt"
WEIGHTS_DEFECT = "./best.pt"

# ===== 类别名设置 =====
GT_CLASS_LABELS = [
    'Crease Gap', 'Punching Hole', 'Silk Spot', 'Crease', 'Waist Folding',
    'Water Spot', 'Oil Spot', 'Inclusion', 'Oil Spot', 'Rolled Pit', 'Waist Folding'
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

# 加载模型（全局变量，避免重复加载）
print("Loading models...")
model_yolov11 = None
model_defect = None

try:
    if Path(WEIGHTS_YOLOV11).exists():
        model_yolov11 = YOLO(WEIGHTS_YOLOV11)
        print("✓ YOLOv11 model loaded")
    else:
        print("✗ YOLOv11 weights not found")
except Exception as e:
    print(f"✗ Error loading YOLOv11: {e}")

try:
    if Path(WEIGHTS_DEFECT).exists():
        model_defect = YOLO(WEIGHTS_DEFECT)
        print("✓ Defect-YOLOv11 model loaded")
    else:
        print("✗ Defect-YOLOv11 weights not found")
except Exception as e:
    print(f"✗ Error loading Defect-YOLOv11: {e}")

def get_available_images():
    """获取可用的图像列表"""
    if not IMG_DIR.exists():
        return []
    images = sorted([f.stem for f in IMG_DIR.glob("*.jpg")])
    return images

def process_image(image_name, conf_threshold, iou_threshold, show_gt, show_yolov11, show_defect):
    """处理图像并生成可视化结果"""
    
    if not image_name:
        return None, None, None, "请选择一个图像"
    
    img_path = IMG_DIR / f"{image_name}.jpg"
    gt_path = GT_TXT_DIR / f"{image_name}.txt"
    
    # 读取原始图像
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None, f"无法加载图像: {img_path}"
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    results = []
    info_text = f"图像: {image_name}\n分辨率: {W}x{H}\n"
    
    # Ground Truth
    gt_img = None
    if show_gt:
        gt_xyxy, gt_cls = read_gt_yolo_to_xyxy(gt_path, W, H)
        gt_vis = draw_boxes(img, gt_xyxy, gt_cls, GT_CLASS_LABELS)
        gt_img = cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB)
        info_text += f"\nGround Truth: {len(gt_xyxy)} 个标注"
    
    # YOLOv11
    yolo_img = None
    if show_yolov11 and model_yolov11 is not None:
        r1 = model_yolov11.predict(source=img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        if r1.boxes is not None:
            xyxy1 = r1.boxes.xyxy.cpu().numpy()
            cls1 = r1.boxes.cls.cpu().numpy().astype(int)
            conf1 = r1.boxes.conf.cpu().numpy()
        else:
            xyxy1 = np.zeros((0, 4), dtype=np.float32)
            cls1 = np.array([], dtype=np.int32)
            conf1 = np.array([], dtype=np.float32)
        
        yolo_vis = draw_boxes(img, xyxy1, cls1, PRED_CLASS_LABELS, conf1)
        yolo_img = cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB)
        info_text += f"\nYOLOv11: {len(xyxy1)} 个检测"
    
    # Defect-YOLOv11
    defect_img = None
    if show_defect and model_defect is not None:
        r2 = model_defect.predict(source=img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        if r2.boxes is not None:
            xyxy2 = r2.boxes.xyxy.cpu().numpy()
            cls2 = r2.boxes.cls.cpu().numpy().astype(int)
            conf2 = r2.boxes.conf.cpu().numpy()
        else:
            xyxy2 = np.zeros((0, 4), dtype=np.float32)
            cls2 = np.array([], dtype=np.int32)
            conf2 = np.array([], dtype=np.float32)
        
        defect_vis = draw_boxes(img, xyxy2, cls2, PRED_CLASS_LABELS, conf2)
        defect_img = cv2.cvtColor(defect_vis, cv2.COLOR_BGR2RGB)
        info_text += f"\nDefect-YOLOv11: {len(xyxy2)} 个检测"
    
    info_text += f"\n\n置信度阈值: {conf_threshold}\nIoU阈值: {iou_threshold}"
    
    return gt_img, yolo_img, defect_img, info_text

# 创建Gradio界面
with gr.Blocks(title="缺陷检测可视化系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 缺陷检测可视化系统")
    gr.Markdown("比较Ground Truth、YOLOv11和Defect-YOLOv11的检测结果")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 控制面板")
            
            # 图像选择
            available_images = get_available_images()
            image_dropdown = gr.Dropdown(
                choices=available_images,
                label="选择图像",
                value=available_images[0] if available_images else None,
                interactive=True
            )
            
            # 参数设置
            conf_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="置信度阈值",
                info="设置检测的置信度阈值"
            )
            
            iou_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.45,
                step=0.05,
                label="NMS IoU阈值",
                info="设置非极大值抑制的IoU阈值"
            )
            
            # 显示选项
            gr.Markdown("### 显示选项")
            show_gt = gr.Checkbox(label="显示Ground Truth", value=True)
            show_yolov11 = gr.Checkbox(label="显示YOLOv11", value=True)
            show_defect = gr.Checkbox(label="显示Defect-YOLOv11", value=True)
            
            # 处理按钮
            process_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")
            
            # 信息显示
            info_box = gr.Textbox(
                label="检测信息",
                lines=10,
                interactive=False
            )
    
    # 结果显示区域
    with gr.Row():
        gt_output = gr.Image(label="Ground Truth", type="numpy")
        yolo_output = gr.Image(label="YOLOv11", type="numpy")
        defect_output = gr.Image(label="Defect-YOLOv11", type="numpy")
    
    # 绑定事件
    process_btn.click(
        fn=process_image,
        inputs=[image_dropdown, conf_slider, iou_slider, show_gt, show_yolov11, show_defect],
        outputs=[gt_output, yolo_output, defect_output, info_box]
    )
    
    # 添加示例
    gr.Markdown("### 📝 使用说明")
    gr.Markdown("""
    1. 从下拉菜单中选择要检测的图像
    2. 调整置信度阈值和IoU阈值（可选）
    3. 选择要显示的结果类型
    4. 点击"开始检测"按钮
    5. 查看并比较三种结果
    
    **提示**: 
    - 较低的置信度阈值会检测出更多目标（但可能有误检）
    - 较高的IoU阈值会保留更多重叠的检测框
    """)

# 启动应用
if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7860, max_attempts=10):
        """查找可用端口"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None
    
    # 查找可用端口
    port = find_free_port()
    if port is None:
        print("❌ 错误：无法找到可用端口")
        exit(1)
    
    print(f"\n{'='*50}")
    print(f"🚀 启动成功！")
    print(f"{'='*50}")
    print(f"📍 本地访问: http://localhost:{port}")
    print(f"📍 局域网访问: http://127.0.0.1:{port}")
    print(f"\n💡 提示: 按 Ctrl+C 停止应用")
    print(f"{'='*50}\n")
    
    try:
        demo.launch(
            server_name="127.0.0.1",  # 改为127.0.0.1，更稳定
            server_port=port,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True  # 自动打开浏览器
        )
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n尝试以下解决方案:")
        print("1. 检查端口是否被占用: lsof -i :7860")
        print("2. 尝试手动指定端口: python visualization_app.py --port 8080")
        print("3. 检查防火墙设置")
        exit(1)