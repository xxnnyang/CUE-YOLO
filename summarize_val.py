import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import argparse, re, json, csv, time
from typing import Dict, Any, List, Tuple, Optional

# 可选：测 FPS 时用到（仅当 val 产物里没有速度信息时才会用）
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# 你想在论文里显示的“方法名”映射（模型名里含这些关键字→映射成固定行名）
# 可按需增删：左侧是正则（忽略大小写），右侧是展示名
DISPLAY_NAME_MAP = [
    (r"simam", "SimAM"),
    (r"local[-_ ]?window.*att", "LocalWindowAttention"),
    (r"triplet.*att", "TripletAttention"),
    (r"\b(cpca|mpca)\b", "CPCA/MPCA"),
    (r"segnext.*att", "SegNext_Attention"),
    (r"cafm", "CAFM"),
    (r"\bmlca\b", "MLCA"),
    (r"bilevel.*routing.*att", "BiLevelRoutingAttention"),
    (r"\barf\b", "ARF (本文)"),
]

POSSIBLE_MAP50_KEYS = ["metrics/mAP50", "metrics/mAP50(B)", "mAP50", "map50"]
POSSIBLE_MAP5095_KEYS = ["metrics/mAP50-95", "metrics/mAP50-95(B)", "mAP50-95", "map"]
POSSIBLE_PARAMS_KEYS = ["model/parameters", "model/params", "params", "parameters"]
POSSIBLE_GFLOPs_KEYS = ["model/GFLOPs", "GFLOPs", "gflops"]
# 常见速度字段（毫秒/张），若拿到其中之一，就能换算 FPS=1000/ms
POSSIBLE_INFER_MS_KEYS = [
    ("speed", "inference"),  # metrics.speed['inference']
    ("speed", "pred"),       # 有些版本用 'pred'
    ("inference",),          # 直接 'inference'
    ("t_inference",),        # 直接 't_inference'
]

def regex_display_name(name: str) -> str:
    n = name.lower()
    for pat, disp in DISPLAY_NAME_MAP:
        if re.search(pat, n, flags=re.I):
            return disp
    return name  # 没命中就用原名

def load_metrics_json(val_dir: Path) -> Optional[Dict[str, Any]]:
    """
    在 val 输出目录里尝试读取 metrics（我们在 batch_val.py 已经落了 metrics.json）
    兜底尝试 results.json / predictions.json
    """
    for fn in ["metrics.json", "results.json", "predictions.json"]:
        p = val_dir / fn
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    # 有些版本把 metrics 存在 runs/val/exp/args.yaml 里，但那通常是参数，不含成绩
    return None

def extract_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def extract_infer_ms(d: Dict[str, Any]) -> Optional[float]:
    """从 metrics 中挖 '单张推理毫秒'"""
    # 形式1：{"speed": {"inference": 1.23, ...}}
    for path in POSSIBLE_INFER_MS_KEYS:
        cur = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    # 形式2：部分版本用 'speed' 单位是 imgs/s，或 'fps'
    for k in ["fps", "FPS", "imgs/s", "imgs_per_second"]:
        if k in d and isinstance(d[k], (int, float)) and d[k] > 0:
            return 1000.0 / (1000.0 / float(d[k]))  # 把 FPS 反推成 ms，再走统一换算
    return None

def find_best_weight(train_root: Path, model_name: str) -> Optional[Path]:
    """
    根据 val 子目录名（GC-DET-{name}）或 {name} 反查训练目录下的权重
    """
    # 常见结构：./summer/runs/train/{name}/weights/best.pt
    cand = train_root / model_name / "weights" / "best.pt"
    if cand.exists():
        return cand
    # 再试试带 GC-DET 前缀
    cand2 = train_root / f"GC-DET-{model_name}" / "weights" / "best.pt"
    if cand2.exists():
        return cand2
    # 深度递归兜底搜（较慢，仅在必要时使用）
    for p in train_root.rglob("best.pt"):
        if p.parent.parent.name.lower() in [model_name.lower(), f"gc-det-{model_name}".lower()]:
            return p
    return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def measure_fps_with_predict(weight: Path, source: Path, imgsz: int = 640, warmup: int = 5, iters: int = 20) -> Optional[float]:
    """
    当 metrics 里没有速度字段时，用 model.predict() 在少量图片上测一个近似 FPS。
    - 仅在 YOLO 可用且有权重、有图片时才会触发
    - 为避免太慢：默认预热5次、计时20次
    """
    if YOLO is None or not weight.exists():
        return None
    # 收集若干张图片
    pics = []
    if source.is_file():
        pics = [str(source)]
    elif source.is_dir():
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
            pics.extend([str(p) for p in source.glob(ext)])
            if len(pics) >= 32:
                break
    if not pics:
        return None

    model = YOLO(str(weight))
    # 预热
    for _ in range(warmup):
        _ = model.predict(source=pics[:4], imgsz=imgsz, verbose=False, device=None, stream=False)
    # 计时
    n_imgs = 0
    t0 = time.time()
    for _ in range(iters):
        res = model.predict(source=pics[:4], imgsz=imgsz, verbose=False, device=None, stream=False)
        # 统计本轮处理的图片数
        # （res 可能是一个列表，每张图一个结果）
        try:
            n_imgs += sum(len(r) > -1 for r in res)  # 粗略按张数累计
        except Exception:
            n_imgs += len(pics[:4])
    t1 = time.time()
    elapsed = t1 - t0
    if elapsed <= 0 or n_imgs == 0:
        return None
    fps = n_imgs / elapsed
    return float(fps)

def parse_args():
    ap = argparse.ArgumentParser("Summarize YOLO val metrics into CSV")
    ap.add_argument("--val-root", type=str, default="./summer/runs/val", help="验证输出根目录（包含多个 GC-DET-*/）")
    ap.add_argument("--train-root", type=str, default="./summer/runs/train", help="训练输出根目录（用于回查 best.pt 以测 FPS/统计 Params/GFLOPs）")
    ap.add_argument("--out", type=str, default="./summer/runs/val/summary.csv", help="CSV 输出路径")
    ap.add_argument("--imgsz", type=int, default=640, help="统计 GFLOPs 时假设的输入尺寸")
    ap.add_argument("--only", nargs="*", default=None, help="仅统计匹配这些正则的模型名")
    ap.add_argument("--exclude", nargs="*", default=None, help="排除匹配这些正则的模型名")
    ap.add_argument("--fps-source", type=str, default="./dataset/GC-DET-new/valid/images", help="当没有速度信息时，用这批图片近似测 FPS（少量采样、耗时很小）")
    return ap.parse_args()

def name_filter(name: str, only: List[str], exclude: List[str]) -> bool:
    if only and not any(re.search(p, name, flags=re.I) for p in only):
        return False
    if exclude and any(re.search(p, name, flags=re.I) for p in exclude):
        return False
    return True

def main():
    args = parse_args()
    val_root = Path(args.val_root)
    train_root = Path(args.train_root)
    fps_source = Path(args.fps_source)

    rows: List[Dict[str, Any]] = []
    # 遍历所有 GC-DET-* 子目录
    for sub in sorted(val_root.glob("GC-DET-*")):
        name = sub.name.replace("GC-DET-", "")
        if not name_filter(name, args.only or [], args.exclude or []):
            continue

        metrics = load_metrics_json(sub) or {}
        # 取 mAP 指标
        map50 = safe_float(extract_first(metrics, POSSIBLE_MAP50_KEYS))
        map5095 = safe_float(extract_first(metrics, POSSIBLE_MAP5095_KEYS))
        # 取 Params/GFLOPs
        params = safe_float(extract_first(metrics, POSSIBLE_PARAMS_KEYS))
        gflops = safe_float(extract_first(metrics, POSSIBLE_GFLOPs_KEYS))
        # 取 FPS（优先从 metrics 中解析）
        infer_ms = extract_infer_ms(metrics)  # 单张毫秒
        fps = None
        if infer_ms and infer_ms > 0:
            fps = 1000.0 / float(infer_ms)
        # 如果没有速度信息，尝试回查权重，做一个轻量级近似测试
        if fps is None:
            weight = find_best_weight(train_root, name)
            if weight:
                fps = measure_fps_with_predict(weight, fps_source, imgsz=args.imgsz)

        row = {
            "模型": regex_display_name(name),
            "mAP@50": f"{map50:.4f}" if map50 is not None else "",
            "mAP@50-95": f"{map5095:.4f}" if map5095 is not None else "",
            "FPS": f"{fps:.2f}" if fps is not None else "",
            "Params": f"{params:.2f}M" if params is not None and params > 1e6 else (f"{params:.0f}" if params else ""),
            "GFLOPs": f"{gflops:.2f}" if gflops is not None else "",
            "原始模型名": name,
            "结果目录": str(sub),
        }
        rows.append(row)

    # 按论文常见排序：先映射名（SimAM…ARF），再把剩余的拼后面
    preferred_order = [x[1] for x in DISPLAY_NAME_MAP]
    def sort_key(r):
        n = r["模型"]
        return (preferred_order.index(n) if n in preferred_order else 999, n.lower())
    rows.sort(key=sort_key)

    # 写 CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = ["模型", "mAP@50", "mAP@50-95", "FPS", "Params", "GFLOPs", "原始模型名", "结果目录"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] 汇总完成：{out}")
    if not rows:
        print("[INFO] 没找到任何可汇总的结果，请检查 --val-root 是否包含 GC-DET-* 子目录。")

if __name__ == "__main__":
    main()
