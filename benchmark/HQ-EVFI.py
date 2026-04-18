import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

"""==========import from our code=========="""
sys.path.append(".")
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab


def parse_metrics(metrics_str):
    aliases = {
        "psnr": "psnr",
        "ssim": "ssim",
        "lpips": "lpips",
        "niqe": "niqe",
        "clip-iqa": "clip-iqa",
        "clip_iqa": "clip-iqa",
        "clipiqa": "clip-iqa",
    }
    metrics = []
    for item in metrics_str.split(","):
        key = item.strip().lower()
        if key == "":
            continue
        if key not in aliases:
            raise ValueError(
                "Unsupported metric '{}'. Supported: psnr,ssim,lpips,niqe,clip-iqa".format(
                    item
                )
            )
        canonical = aliases[key]
        if canonical not in metrics:
            metrics.append(canonical)
    if len(metrics) == 0:
        raise ValueError("At least one metric must be selected.")
    return metrics


def find_sequences(root_path):
    seq_dirs = []
    seen = set()

    # Case 1: single sequence folder directly: root/images/*.png
    direct_img_dir = os.path.join(root_path, "visual_RGB")
    if os.path.isdir(direct_img_dir):
        seq_name = os.path.basename(os.path.normpath(root_path))
        seq_dirs.append((seq_name, direct_img_dir))
        return seq_dirs

    # Case 2: any nested sequence folder containing an "images" subfolder.
    # This supports layouts like:
    # root/1_TEST/acquarium_08/images/*.png
    for current_root, dirnames, _ in os.walk(root_path):
        if "visual_RGB" in dirnames:
            img_dir = os.path.join(current_root, "visual_RGB")
            key = os.path.normpath(img_dir)
            if key not in seen:
                seen.add(key)
                seq_name = os.path.basename(os.path.normpath(current_root))
                seq_dirs.append((seq_name, img_dir))

    # Case 3: folders that store frames directly without "images" subfolder.
    # We consider a directory a sequence if it has at least one image file.
    if len(seq_dirs) == 0:
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for current_root, _, filenames in os.walk(root_path):
            has_image = any(
                os.path.splitext(f)[1].lower() in valid_exts for f in filenames
            )
            if not has_image:
                continue
            key = os.path.normpath(current_root)
            if key in seen:
                continue
            seen.add(key)
            seq_name = os.path.basename(key)
            seq_dirs.append((seq_name, current_root))

    seq_dirs.sort(key=lambda x: x[0])
    return seq_dirs


def list_frames(img_dir, exts):
    names = []
    for n in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(n)[1].lower().lstrip(".")
        if ext in exts:
            names.append(os.path.join(img_dir, n))
    return names


def bgr_uint8_to_cuda_tensor(img):
    return (
        torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0
    )


def to_rgb_minus1_1(tensor_bgr_01):
    tensor_rgb_01 = tensor_bgr_01[:, [2, 1, 0], :, :]
    return tensor_rgb_01 * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ours", type=str)
    parser.add_argument("--path", type=str, required=True, help="Root of BS-ERGB dataset")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Directory to save predicted frames, e.g. G:/val/output. If unset, no images are saved.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="psnr,ssim,lpips,niqe,clip-iqa",
        help="Comma-separated metrics: psnr,ssim,lpips,niqe,clip-iqa",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="png,jpg,jpeg,bmp",
        help="Comma-separated image extensions under each sequence/images",
    )
    args = parser.parse_args()

    assert args.model in ["ours", "ours_small"], "Model not exists!"
    selected_metrics = parse_metrics(args.metrics)
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip() != ""}
    save_outputs = args.output_root is not None and args.output_root.strip() != ""
    output_root = os.path.normpath(args.output_root) if save_outputs else None
    if save_outputs:
        os.makedirs(output_root, exist_ok=True)

    """==========Model setting=========="""
    TTA = True
    if args.model == "ours_small":
        TTA = False
        cfg.MODEL_CONFIG["LOGNAME"] = "ours_small"
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(F=16, depth=[2, 2, 2, 2, 2])
    else:
        cfg.MODEL_CONFIG["LOGNAME"] = "ours"
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()

    # Optional metrics are loaded on demand with explicit dependency messages.
    lpips_metric = None
    niqe_metric = None
    clipiqa_metric = None

    if "lpips" in selected_metrics:
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "LPIPS requested but package 'lpips' is not installed. Run: pip install lpips"
            ) from e
        lpips_metric = lpips.LPIPS(net="alex").cuda().eval()

    if "niqe" in selected_metrics or "clip-iqa" in selected_metrics:
        try:
            import pyiqa
        except ImportError as e:
            raise ImportError(
                "NIQE/CLIP-IQA requested but package 'pyiqa' is not installed. Run: pip install pyiqa"
            ) from e
        if "niqe" in selected_metrics:
            niqe_metric = pyiqa.create_metric("niqe", device="cuda")
        if "clip-iqa" in selected_metrics:
            clipiqa_metric = pyiqa.create_metric("clipiqa", device="cuda")

    print("=========================Starting testing=========================")
    print(
        "Dataset: BS-ERGB   Model: {}   TTA: {}   Metrics: {}   SaveOutput: {}".format(
            model.name, TTA, ",".join(selected_metrics), save_outputs
        )
    )

    seq_dirs = find_sequences(args.path)
    if len(seq_dirs) == 0:
        raise RuntimeError(
            "No valid sequence found. Supported structures include: "
            "root/sequence_name/images/*.png or root/split/sequence_name/images/*.png"
        )

    totals = {m: [] for m in selected_metrics}
    total_samples = 0
    used_sequences = 0

    for seq_name, img_dir in seq_dirs:
        frame_paths = list_frames(img_dir, exts)
        if len(frame_paths) < 3:
            print("[Skip] {} has less than 3 frames".format(seq_name))
            continue
        if save_outputs:
            seq_out_dir = os.path.join(output_root, seq_name)
            os.makedirs(seq_out_dir, exist_ok=True)

        seq_scores = {m: [] for m in selected_metrics}
        triplet_count = len(frame_paths) - 2
        valid_triplets = 0
        pbar = tqdm(range(triplet_count), desc="{}".format(seq_name), leave=False)
        for i in pbar:
            frame0 = cv2.imread(frame_paths[i], cv2.IMREAD_COLOR)
            frame1 = cv2.imread(frame_paths[i + 1], cv2.IMREAD_COLOR)
            frame2 = cv2.imread(frame_paths[i + 2], cv2.IMREAD_COLOR)
            if frame0 is None or frame1 is None or frame2 is None:
                continue
            valid_triplets += 1

            I0 = bgr_uint8_to_cuda_tensor(frame0)
            I1_gt = bgr_uint8_to_cuda_tensor(frame1)
            I2 = bgr_uint8_to_cuda_tensor(frame2)

            padder = InputPadder(I0.shape, divisor=32)
            I0_pad, I2_pad = padder.pad(I0, I2)
            I1_pred = model.inference(I0_pad, I2_pad, TTA=TTA, fast_TTA=TTA)[0]
            I1_pred = padder.unpad(I1_pred).unsqueeze(0).clamp(0.0, 1.0)

            if save_outputs:
                pred_uint8 = (
                    I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
                ).round().clip(0, 255).astype(np.uint8)
                out_name = os.path.basename(frame_paths[i + 1])
                out_path = os.path.join(seq_out_dir, out_name)
                cv2.imwrite(out_path, pred_uint8)

            if "ssim" in selected_metrics:
                ssim = float(ssim_matlab(I1_gt, I1_pred).detach().cpu().numpy())
                seq_scores["ssim"].append(ssim)

            if "psnr" in selected_metrics:
                gt_np = I1_gt[0].detach().cpu().numpy().transpose(1, 2, 0)
                pred_np = I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0)
                pred_np = np.round(pred_np * 255.0) / 255.0
                psnr = -10 * math.log10(((gt_np - pred_np) * (gt_np - pred_np)).mean())
                seq_scores["psnr"].append(psnr)

            if "lpips" in selected_metrics:
                with torch.no_grad():
                    lp = lpips_metric(to_rgb_minus1_1(I1_gt), to_rgb_minus1_1(I1_pred))
                seq_scores["lpips"].append(float(lp.item()))

            if "niqe" in selected_metrics:
                with torch.no_grad():
                    ni = niqe_metric(I1_pred)
                seq_scores["niqe"].append(float(ni.item()))

            if "clip-iqa" in selected_metrics:
                with torch.no_grad():
                    cq = clipiqa_metric(I1_pred)
                seq_scores["clip-iqa"].append(float(cq.item()))

        parts = []
        for m in selected_metrics:
            if len(seq_scores[m]) == 0:
                continue
            seq_mean = float(np.mean(seq_scores[m]))
            totals[m].extend(seq_scores[m])
            parts.append("{}: {:.6f}".format(m.upper(), seq_mean))
        if len(parts) > 0:
            used_sequences += 1
            total_samples += valid_triplets
            print("[{}] {}".format(seq_name, " | ".join(parts)))

    print("-------------------------Overall-------------------------")
    for m in selected_metrics:
        if len(totals[m]) == 0:
            continue
        print("Avg {}: {:.6f}".format(m.upper(), float(np.mean(totals[m]))))
    print("Evaluated sequences: {} | Triplets: {}".format(used_sequences, total_samples))


if __name__ == "__main__":
    main()
