import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from torchvision.utils import flow_to_image

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
    direct_img_dir = os.path.join(root_path, "images")
    if os.path.isdir(direct_img_dir):
        seq_name = os.path.basename(os.path.normpath(root_path))
        seq_dirs.append((seq_name, direct_img_dir))
        return seq_dirs

    # Case 2: any nested sequence folder containing an "images" subfolder.
    # This supports layouts like:
    # root/1_TEST/acquarium_08/images/*.png
    for current_root, dirnames, _ in os.walk(root_path):
        if "images" in dirnames:
            img_dir = os.path.join(current_root, "images")
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


def build_model(args):
    assert args.model in ["ours", "ours_small"], "Model not exists!"
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
    return model, TTA


def load_metrics(selected_metrics):
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

    metric_fns = {
        "ssim": lambda gt, pred: float(ssim_matlab(gt, pred).detach().cpu().numpy()),
        "psnr": lambda gt, pred: (
            -10
            * math.log10(
                (
                    (
                        gt[0].detach().cpu().numpy().transpose(1, 2, 0)
                        - np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
                        / 255.0
                    )
                    ** 2
                ).mean()
            )
        ),
        "lpips": lambda gt, pred: float(
            lpips_metric(to_rgb_minus1_1(gt), to_rgb_minus1_1(pred)).item()
        ),
        "niqe": lambda gt, pred: float(niqe_metric(pred).item()),
        "clip-iqa": lambda gt, pred: float(clipiqa_metric(pred).item()),
    }
    return {k: metric_fns[k] for k in selected_metrics}


def process_triplet(model, frame0, frame1, frame2, TTA, mode):
    I0 = bgr_uint8_to_cuda_tensor(frame0)
    I1_gt = bgr_uint8_to_cuda_tensor(frame1)
    I2 = bgr_uint8_to_cuda_tensor(frame2)

    padder = InputPadder(I0.shape, divisor=32)
    I0_pad, I2_pad = padder.pad(I0, I2)

    final_flow = None
    if mode == "flow_vis":
        model_input = torch.cat((I0_pad, I2_pad), 1)
        flow_list, _, _, I1_pred = model.net(model_input, timestep=0.5)
        final_flow = padder.unpad(flow_list[-1])
    else:
        I1_pred = model.inference(I0_pad, I2_pad, TTA=TTA, fast_TTA=TTA)[0]

    I1_pred = padder.unpad(I1_pred).unsqueeze(0).clamp(0.0, 1.0)
    return I1_gt, I1_pred, final_flow


def save_pred_frame(pred, out_path):
    pred_uint8 = (
        pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, pred_uint8)


def save_flow_vis(final_flow, seq_out_dir, frame_idx):
    flow_0 = final_flow[:, :2]
    flow_1 = final_flow[:, 2:4]

    flow_img_0 = flow_to_image(flow_0)[0].detach().cpu().numpy().transpose(1, 2, 0)
    flow_img_1 = flow_to_image(flow_1)[0].detach().cpu().numpy().transpose(1, 2, 0)

    flow_img_0_bgr = cv2.cvtColor(flow_img_0, cv2.COLOR_RGB2BGR)
    flow_img_1_bgr = cv2.cvtColor(flow_img_1, cv2.COLOR_RGB2BGR)

    name_0 = "{:04d}_flow0.png".format(frame_idx)
    name_1 = "{:04d}_flow1.png".format(frame_idx)
    cv2.imwrite(os.path.join(seq_out_dir, name_0), flow_img_0_bgr)
    cv2.imwrite(os.path.join(seq_out_dir, name_1), flow_img_1_bgr)


def run_eval(args, model, TTA, exts):
    selected_metrics = parse_metrics(args.metrics)
    metric_fns = load_metrics(selected_metrics)

    save_outputs = args.output_root is not None and args.output_root.strip() != ""
    output_root = os.path.normpath(args.output_root) if save_outputs else None
    if save_outputs:
        os.makedirs(output_root, exist_ok=True)

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

            I1_gt, I1_pred, _ = process_triplet(
                model=model, frame0=frame0, frame1=frame1, frame2=frame2, TTA=TTA, mode="eval"
            )

            if save_outputs:
                out_name = os.path.basename(frame_paths[i + 1])
                save_pred_frame(I1_pred, os.path.join(seq_out_dir, out_name))

            for metric_name, metric_fn in metric_fns.items():
                with torch.no_grad():
                    value = metric_fn(I1_gt, I1_pred)
                seq_scores[metric_name].append(value)

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


def run_flow_vis(args, model, TTA, exts):
    save_outputs = args.save_flow_vis
    if not save_outputs:
        print("[Info] --save_flow_vis is not set. Flow visualization files will not be saved.")

    output_root = os.path.normpath(args.output_root) if args.output_root else None
    if save_outputs and output_root is None:
        raise ValueError("--output_root is required when --mode flow_vis and --save_flow_vis are used.")
    if save_outputs:
        os.makedirs(output_root, exist_ok=True)

    print("======================Starting flow_vis======================")
    print(
        "Dataset: BS-ERGB   Model: {}   TTA: {}   SaveFlowVis: {}   MaxN: {}".format(
            model.name, TTA, save_outputs, args.flow_vis_max_n
        )
    )

    seq_dirs = find_sequences(args.path)
    if len(seq_dirs) == 0:
        raise RuntimeError(
            "No valid sequence found. Supported structures include: "
            "root/sequence_name/images/*.png or root/split/sequence_name/images/*.png"
        )

    total_saved = 0
    processed_sequences = 0

    for seq_name, img_dir in seq_dirs:
        frame_paths = list_frames(img_dir, exts)
        if len(frame_paths) < 3:
            print("[Skip] {} has less than 3 frames".format(seq_name))
            continue

        processed_sequences += 1
        seq_out_dir = None
        if save_outputs:
            seq_out_dir = os.path.join(output_root, seq_name)
            os.makedirs(seq_out_dir, exist_ok=True)

        triplet_count = len(frame_paths) - 2
        max_n = min(args.flow_vis_max_n, triplet_count)
        pbar = tqdm(range(max_n), desc="{}".format(seq_name), leave=False)
        saved_in_seq = 0

        for i in pbar:
            frame0 = cv2.imread(frame_paths[i], cv2.IMREAD_COLOR)
            frame1 = cv2.imread(frame_paths[i + 1], cv2.IMREAD_COLOR)
            frame2 = cv2.imread(frame_paths[i + 2], cv2.IMREAD_COLOR)
            if frame0 is None or frame1 is None or frame2 is None:
                continue

            _, _, final_flow = process_triplet(
                model=model, frame0=frame0, frame1=frame1, frame2=frame2, TTA=TTA, mode="flow_vis"
            )
            if save_outputs:
                save_flow_vis(final_flow, seq_out_dir, i + 1)
                saved_in_seq += 1

        total_saved += saved_in_seq
        print("[{}] flow visualized: {}".format(seq_name, saved_in_seq))

    print("-------------------------FlowVis-------------------------")
    print("Processed sequences: {} | Saved flow pairs: {}".format(processed_sequences, total_saved))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ours", type=str)
    parser.add_argument("--path", type=str, required=True, help="Root of BS-ERGB dataset")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory. In eval mode: predicted frames. In flow_vis mode: flow visualization PNGs.",
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
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "flow_vis"],
        help="Execution mode: eval (default) or flow_vis.",
    )
    parser.add_argument(
        "--flow_vis_max_n",
        type=int,
        default=30,
        help="Number of triplets to visualize per sequence in flow_vis mode.",
    )
    parser.add_argument(
        "--save_flow_vis",
        action="store_true",
        help="Whether to save flow visualization images in flow_vis mode.",
    )
    args = parser.parse_args()

    if args.flow_vis_max_n < 0:
        raise ValueError("--flow_vis_max_n must be >= 0")

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip() != ""}
    model, TTA = build_model(args)

    if args.mode == "eval":
        run_eval(args, model, TTA, exts)
        return

    run_flow_vis(args, model, TTA, exts)


if __name__ == "__main__":
    main()
