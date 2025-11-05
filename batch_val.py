import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mmengine")

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import re
import json

def find_models(train_root: Path):
    """在 train_root 下寻找 **/weights/best.pt，返回 [(name, weight_path)]"""
    pairs = []
    for best in train_root.glob("*/weights/best.pt"):
        name = best.parent.parent.name  # .../train/{name}/weights/best.pt
        pairs.append((name, best))
    # 也兼容更深层目录：.../train/xxx/yyy/weights/best.pt
    for best in train_root.glob("**/weights/best.pt"):
        name = best.parent.parent.name
        if (name, best) not in pairs:
            pairs.append((name, best))
    # 去重并排序
    uniq = {}
    for n, p in pairs:
        uniq[str(p)] = (n, p)
    pairs = sorted(uniq.values(), key=lambda x: x[0].lower())
    return pairs

def should_skip(val_project: Path, exp_name: str, reval: bool):
    """存在结果时是否跳过。这里以生成的目录是否存在为准，也尝试检查 metrics 文件。"""
    out_dir = val_project / exp_name
    if reval:
        return False
    if not out_dir.exists():
        return False
    # 常见产物：results.json 或 metrics.json（不同版本命名可能不同）
    possible = [
        out_dir / "results.json",
        out_dir / "metrics.json",
        out_dir / "predictions.json",
        out_dir / "val_batch0_labels.jpg",  # 有图基本说明跑过
    ]
    return any(p.exists() for p in possible)

def name_match(name: str, only: list[str], exclude: list[str]) -> bool:
    """基于正则包含/排除过滤"""
    if only:
        ok = any(re.search(pat, name) for pat in only)
        if not ok:
            return False
    if exclude:
        if any(re.search(pat, name) for pat in exclude):
            return False
    return True

def dump_metrics(metrics_obj, out_file: Path):
    """尽量把 metrics 对象转成可读 JSON 存档（兼容不同版本Ultralytics）"""
    try:
        if hasattr(metrics_obj, "results_dict"):
            data = metrics_obj.results_dict
        elif hasattr(metrics_obj, "keys") and hasattr(metrics_obj, "values"):
            data = dict(zip(list(metrics_obj.keys()), list(metrics_obj.values())))
        elif isinstance(metrics_obj, dict):
            data = metrics_obj
        else:
            # 最后兜底：尝试 __dict__
            data = getattr(metrics_obj, "__dict__", {"_repr": str(metrics_obj)})
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        out_file.write_text(json.dumps({"error": f"dump metrics failed: {e}"}, ensure_ascii=False, indent=2))


# python batch_val.py --train-root ./summer/runs/train/NEU-DET-flip --data ./dataset/NEU-DET-flip/data.yaml --reval --split test --val-project ./summer/runs/test/NEU-DET-flip
# python batch_val.py --train-root ./summer/runs/train/GC-DET-new --data ./dataset/GC-DET-new/data.yaml --reval --split val --val-project ./summer/runs/val/GC-DET-new

def main():
    parser = argparse.ArgumentParser(description="Auto-scan and batch validate YOLO models.")
    parser.add_argument("--train-root", type=str, default="./summer/runs/train", help="训练权重根目录（会递归搜寻 */weights/best.pt）")
    parser.add_argument("--val-project", type=str, default="./summer/runs/val", help="验证输出根目录")
    parser.add_argument("--data", type=str, default="./dataset/GC-DET-new/data.yaml", help="data.yaml 路径")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="数据划分")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--reval", action="store_true", help="无论是否存在结果都重新验证")
    parser.add_argument("--only", nargs="*", default=None, help="仅验证匹配这些正则的模型名（空格分隔多个）")
    parser.add_argument("--exclude", nargs="*", default=None, help="排除匹配这些正则的模型名")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile（某些环境不支持时可加）")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    val_project = Path(args.val_project)
    val_project.mkdir(parents=True, exist_ok=True)

    pairs = find_models(train_root)
    if not pairs:
        print(f"[WARN] 在 {train_root} 下没有找到任何 best.pt")
        return

    # 过滤
    pairs = [(n, p) for (n, p) in pairs if name_match(n, args.only or [], args.exclude or [])]
    if not pairs:
        print("[INFO] 过滤后无模型可验证。")
        return

    print(f"[INFO] 共发现 {len(pairs)} 个模型待处理。")

    summary = []
    for name, weight in pairs:
        exp_name = f"{name}"
        if should_skip(val_project, exp_name, args.reval):
            print(f"[SKIP] 已存在结果，跳过：{name}")
            continue

        print(f"\n>>> 开始验证：{name}\n    权重：{weight}\n    输出：{val_project / exp_name}")
        try:
            model = YOLO(str(weight))
            if not args.no_compile:
                try:
                    model = torch.compile(model)
                except Exception as ce:
                    print(f"[WARN] torch.compile 失败，将以未编译模型继续：{ce}")

            metrics = model.val(
                data=args.data,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                plots=True,
                save_json=True,
                project=str(val_project),
                name=exp_name,
            )

            # 存一份 metrics.json 方便统一读取
            dump_metrics(metrics, val_project / exp_name / "metrics.json")
            # 简要提取常用指标
            row = {"name": name}
            try:
                d = getattr(metrics, "results_dict", None) or {}
                # 常见字段（不同版本可能有差异，尽量兼容）
                for k in ["metrics/mAP50", "metrics/mAP50-95", "metrics/precision(B)", "metrics/recall(B)"]:
                    if k in d:
                        row[k] = d[k]
            except Exception:
                pass
            summary.append(row)

        except Exception as e:
            print(f"[ERROR] 验证失败：{name} -> {e}")
            summary.append({"name": name, "error": str(e)})

    # 输出一个总表
    if summary:
        out = val_project / "batch_val_summary.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n[OK] 批量完成。汇总已写入：{out}")
    else:
        print("\n[INFO] 没有生成任何汇总（可能全部被跳过）。")

if __name__ == "__main__":
    main()
