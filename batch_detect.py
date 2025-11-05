import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
import re
from ultralytics import YOLO

# ---- utils ----
def find_best_pts():
    """
    递归搜寻两种常见训练根目录下的 best.pt：
      1) ./summer/runs/train/**/weights/best.pt
      2) ./runs/runs/train/**/weights/best.pt
    返回 [(name, best_path)]，name 会尽量去掉 'GC-DET-' 前缀
    """
    roots = [Path("./summer/runs/train"), Path("./runs/runs/train")]
    seen = {}
    for root in roots:
        if not root.exists():
            continue
        for best in root.glob("**/weights/best.pt"):
            # 模型名 = best.pt 所在的 {name} 目录
            raw_name = best.parent.parent.name  # .../{name}/weights/best.pt
            name = raw_name
            if raw_name.startswith("GC-DET-"):
                name = raw_name[len("GC-DET-"):]
            key = str(best.resolve())
            seen[key] = (name, best)
    pairs = sorted(seen.values(), key=lambda x: x[0].lower())
    return pairs

def name_match(name: str, only, exclude) -> bool:
    if only:
        if not any(re.search(p, name) for p in only):
            return False
    if exclude:
        if any(re.search(p, name) for p in exclude):
            return False
    return True

def should_skip(project: Path, name: str, overwrite: bool) -> bool:
    if overwrite:
        return False
    out_dir = project / name
    return out_dir.exists() and any(
        (out_dir / p).exists() for p in [
            "predictions.json",
            "labels",         # save_txt=True 时会有
            "crops",          # save_crop=True 时会有
            "image0.jpg"      # 部分版本会直接落推理图
        ]
    )

# ---- main ----

# python batch_detect.py 
def main():
    ap = argparse.ArgumentParser("Batch YOLO detect (predict)")
    ap.add_argument("--source", type=str, default="./dataset/GC-DET-new/valid/images",
                    help="推理数据源(文件/文件夹/通配符/视频/摄像头等)")
    ap.add_argument("--project", type=str, default="./runs/detect",
                    help="输出根目录")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--device", type=str, default=None, help="如 '0' 或 '0,1'")
    ap.add_argument("--half", action="store_true", help="FP16 推理（需设备支持）")
    ap.add_argument("--save-txt", action="store_true", help="保存YOLO txt标签")
    ap.add_argument("--save-conf", action="store_true", help="在txt里保存conf")
    ap.add_argument("--save-crop", action="store_true", help="保存裁剪目标图")
    ap.add_argument("--line-width", type=int, default=None, help="框线宽")
    ap.add_argument("--show-conf", action="store_true", help="显示置信度")
    ap.add_argument("--show-labels", action="store_true", help="显示类别名")
    ap.add_argument("--visualize", action="store_true", help="可视化特征图")
    ap.add_argument("--only", nargs="*", default=None, help="仅跑匹配这些正则的模型名")
    ap.add_argument("--exclude", nargs="*", default=None, help="排除匹配这些正则的模型名")
    ap.add_argument("--overwrite", action="store_true", help="如输出目录存在也强制覆盖/重跑")
    args = ap.parse_args()

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    pairs = find_best_pts()
    if not pairs:
        print("[WARN] 没找到任何 best.pt")
        return

    pairs = [(n, p) for n, p in pairs if name_match(n, args.only, args.exclude)]
    if not pairs:
        print("[INFO] 过滤后无模型可预测。")
        return

    print(f"[INFO] 共发现 {len(pairs)} 个模型待预测。")
    for name, weight in pairs:
        out_name = name  # 每个模型单独的输出子目录
        if should_skip(project, out_name, args.overwrite):
            print(f"[SKIP] 已存在输出，跳过：{name}")
            continue

        print(f"\n>>> 开始预测：{name}\n    权重：{weight}\n    输出：{project/out_name}")
        try:
            model = YOLO(str(weight))
            model.predict(
                source=args.source,
                imgsz=args.imgsz,
                project=str(project),
                name=out_name,
                save=True,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_crop=args.save_crop,
                line_width=args.line_width,
                show_conf=args.show_conf,
                show_labels=args.show_labels,
                visualize=args.visualize,
                exist_ok=args.overwrite,  # 允许覆盖
                agnostic_nms=False,       # 需要可自己改成 True
            )
        except Exception as e:
            print(f"[ERROR] 预测失败：{name} -> {e}")

    print("\n[OK] 批量预测完成。")

if __name__ == "__main__":
    main()