# %% Imports

import os
import torch
import yaml
import numpy as np
import imageio.v2 as imageio
import matplotlib.cm as cm
import models.conv3d_autoencoder as conv3d_autoencoder
from utils.image_utils import to_rgb, upscale_array, upscale_image, percent_norm
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, find_objects

# %% Methods


def evaluate_model(args):
    """
    Warning! this assumes video is (1, T, H, W)
    full workflow for evaluate:
    -> load model (assigning the available device: i.e., cuda or cpu)
    -> load full_video and then slide through with clips of length T=clip_size
    -> run model on each clip and construct recon error heat maps
    -> normalize or clip errors before overlaying
    -> overlay heat map or boundry boxes on original frames
    -> rebuild and possibly annotate video, save
    -> consider means of thresholding "anomalies" from recon error
    """
    # Initialize
    movie_dir = os.path.join(args.data_clips, "full_movie_eval", "full_movie.pt")
    output_path = os.path.join(args.data_clips, "full_movie_eval", "anomaly_video.mp4")
    model_dir = os.path.join(args.model_path, "best_model", "best_model.pth.tar")
    config_path = os.path.join(args.model_path, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  # load config file into local var as hierarchal dict

    clip_len = config["data_pre_processing"]["clip_length"]
    stride = config["evaluate"]["stride"]
    threshold = config["evaluate"]["threshold"]
    min_area = config["evaluate"]["min_area"]
    weight = config["evaluate"]["heat_weight"]
    scale_factor = config["evaluate"]["scale_factor"]
    device = args.device

    # Initialize model
    model_class_name = config["model"]["model_arch"]
    model_class = getattr(conv3d_autoencoder, model_class_name)
    model = model_class()
    model.to(device)

    # Load model & set to eval mode
    checkpoint = torch.load(model_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load full_video to be avaluated by anomaly detector
    video = torch.load(movie_dir)  # shape: (T, H, W) or (1, T, H, W)

    if video.ndim == 3:
        video = video.unsqueeze(0)  # ensure shape is (1, T, H, W)

    C, T, H, W = video.shape

    # Generate sliding clips of video
    clips = []
    for i in range(0, T - clip_len + 1, stride):
        clip = video[:, i : i + clip_len, :, :]
        clips.append(clip)

    # Run model inference per clip & compute error heatmaps
    heatmaps = []  # shape (T, H, W)
    recons = []  # shape (T, H, W)
    for clip in tqdm(clips, desc=f"Reconstructing Clips"):
        input_clip = clip.unsqueeze(0).to(device)  # shape: (1, 1, T, H, W)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=torch.cuda.is_available()
        ):
            recon = model(input_clip)

        error = torch.abs(recon - input_clip).squeeze(0).squeeze(0)  # shape (T, H, W)
        recons.append(recon.squeeze(0).squeeze(0).cpu())
        heatmaps.append(error.cpu())

    # heatmaps = [e.mean()dim=0 for e in heatmaps] # (option) condense each clip into a single map, now list of (H, W)

    # For heatmap aggregation, there are many options with adv & disadv.
    # This arises due to the occurance of each frame accross multiple clip evals when stide < clip_length.
    # Two common approcuhes include:
    # - one heatmap per clip (e.g., masked to the middle of each clip)
    # - one heatmap per frame aggregated (max, min, avg, etc) accross multipl clip model.evals()

    # The former is far more lightweight, but can reduce geenralization while only seeing a heatmap ~1 in every `stride` frame.
    # The latter is more computationaly intensive, but better represents the model as a whole IF correct aggregation used...
    # ... relative to given data. Several examples are provided (max_agg(), mean_agg(), sum_agg()):

    # Max aggregation per frame per pixel:
    final_heat, final_recon = percentile_agg(
        recons, heatmaps, stride, clip_len, T, H, W
    )

    # Normalize heatmap and/or frames?
    # final_heat -= final_heat.min()
    # final_heat /= final_heat.max() + 1e-8  # avoid div-by-zero

    # max = video.max()
    # min = video.min()
    # video = (video - min) / (max - min)

    # Write video for heat_map/boxes overlay via imageio.v2
    video_np = video.squeeze(0).numpy()  # (T, H, W)
    recon_np = final_recon.cpu().numpy()  # (T, H, W)
    heat_np = final_heat.cpu().numpy()  # (T, H, W)
    fps = config["evaluate"]["fps"]

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    if args.eval_diagnostic == False:

        # eval video for anomaly detection only

        for t in tqdm(range(T), desc="Assembling Eval .mp4"):
            frame = video_np[t]
            heat = heat_np[t]

            # percentile normalization per frame based on data, not recon (good for visualizing localised anomalys)
            frame = percent_norm(frame)

            # Apply colour maps
            cmap = cm.get_cmap(
                "magma"
            )  # -> cmap(x) return ([R, G, B, A]) for x in [0,1]
            frame = cmap(frame)[
                :, :, :3
            ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)

            # Ensure all frames are in uint8 RGB
            frame_rgb = to_rgb(frame)

            # Generate upscaled heatmap overlay
            if args.eval_mode == "heatmap":
                anomaly_pil = overlay_heatmap(frame, heat, weight, scale_factor)

            elif args.eval_mode == "boxes":
                anomaly_pil = overlay_heatmap_with_boxes(
                    frame, heat, threshold, min_area, weight, scale_factor
                )
            else:
                raise ValueError(f"Unknown eval mode: {args.eval_mode}")

            # Convert data frame to PIL and upscale
            frame_pil = upscale_image(Image.fromarray(frame_rgb))

            # Compute heatmap stats
            stats = {
                "frame": t,
                "mean": heat.mean(),
                "max": heat.max(),
                "std": heat.std(),
            }

            # Annotate frames
            frame_pil = annotate_frame(frame_pil, "data")
            anomaly_pil = annotate_frame(anomaly_pil, "anomaly", stats=stats)

            # Concatenate horizontally
            row = Image.new("RGB", (frame_pil.width * 2, frame_pil.height))
            row.paste(frame_pil, (0, 0))
            row.paste(anomaly_pil, (frame_pil.width, 0))

            writer.append_data(np.array(row))

    else:

        # eval video with full diagnostic output

        for t in tqdm(range(T), desc="Assembling Eval .mp4"):
            frame = video_np[t]
            heat = heat_np[t]
            recon = recon_np[t]

            # percentile normalization per frame based on data, not recon (good for visualizing localised anomalys)
            frame, recon = percent_norm(frame, recon)

            # Apply colour maps
            cmap = cm.get_cmap(
                "magma"
            )  # -> cmap(x) return ([R, G, B, A]) for x in [0,1]
            frame = cmap(frame)[
                :, :, :3
            ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)
            recon = cmap(recon)[
                :, :, :3
            ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)

            # Ensure all frames are in uint8 RGB
            frame_rgb = to_rgb(frame)
            recon_rgb = to_rgb(recon)

            # Generate heatmap overlay
            if args.eval_mode == "heatmap":
                anomaly_pil = overlay_heatmap(frame, heat, weight, scale_factor)
            elif args.eval_mode == "boxes":
                anomaly_pil = overlay_heatmap_with_boxes(
                    frame, heat, threshold, min_area, weight, scale_factor
                )
            else:
                raise ValueError(f"Unknown eval mode: {args.eval_mode}")

            # Compute heatmap stats
            stats = {
                "frame": t,
                "mean": heat.mean(),
                "max": heat.max(),
                "std": heat.std(),
            }

            # Convert all to PIL and upscale
            frame_pil = upscale_image(Image.fromarray(frame_rgb))
            recon_pil = upscale_image(Image.fromarray(recon_rgb))
            anomaly_pil = upscale_image(anomaly_pil)

            # Annotate frames
            frame_pil = annotate_frame(frame_pil, "data")
            recon_pil = annotate_frame(recon_pil, "recon")
            anomaly_pil = annotate_frame(anomaly_pil, "anomaly", stats=stats)

            # Concatenate horizontally
            row = Image.new("RGB", (frame_pil.width * 3, frame_pil.height))
            row.paste(frame_pil, (0, 0))
            row.paste(recon_pil, (frame_pil.width, 0))
            row.paste(anomaly_pil, (frame_pil.width * 2, 0))

            writer.append_data(np.array(row))

    writer.close()
    print(f"[INFO] Anomaly video with boxes saved to {output_path}")


def overlay_heatmap(frame_grey, heat_map, weight, scale_factor):
    # method assumes to convert float np array to uint8 PIL image for image display
    # frame_grey can be [0,1] or [0,255], shape (H,W); heat_map also (H,W) but only [0,1]

    # percentile normalization per frame (good for visualizing localised anomalys)
    heat_map = percent_norm(heat_map)

    # z_normed per frame (isnt great for visual but good for enhanced post-processing)
    # mean = heat_map.mean()
    # std = heat_map.std()
    # heat_map = (heat_map - mean) / (std + 1e-8)

    frame_rgb = to_rgb(frame_grey)  # convert to uint8 and (H,W)->(H,W,3)

    cmap = cm.get_cmap("jet")  # -> cmap(x) return ([R, G, B, A]) for x in [0,1]

    heat_colored = cmap(heat_map)[
        :, :, :3
    ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)
    heat_colored = to_rgb(heat_colored)

    blended_np = ((1 - weight) * frame_rgb + weight * heat_colored).astype(
        np.uint8
    )  # blends grey scale and heat map, can alter weight as desired
    blended_img = Image.fromarray(blended_np)  # convert np array to PIL image

    return upscale_image(blended_img, scale_factor)


def overlay_heatmap_with_boxes(
    frame_grey, heat_map, threshold, min_area, weight, scale_factor
):
    """Currently includes:
    - area filtering
    - threshold filtering
    - Non-maximum suppresion (IoU, larger area prioritization)"""

    # percentile normalization per frame (good for visualizing localised anomalys)
    heat_map = percent_norm(heat_map)

    frame_grey_up = upscale_array(frame_grey)
    heat_map_up = upscale_array(heat_map)

    frame_rgb = to_rgb(frame_grey_up)  # convert to uint8 and (H,W)->(H,W,3)

    # create anomaly mask and assign labels to regions
    mask = (heat_map_up > threshold).astype(
        np.uint8
    )  # create 0or1 binary image (H,W) of frame past threshold
    labeled_mask, num_labels = label(
        mask
    )  # assign labels to each 'connected region' in the mask (background = 0), (H, W)

    # create heamap colour map overlay
    cmap = cm.get_cmap("jet")
    heat_colored = cmap(heat_map_up)[
        :, :, :3
    ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)
    heat_colored = to_rgb(heat_colored)

    # blend rgb frame and heatmap
    blended_np = ((1 - weight) * frame_rgb + weight * heat_colored).astype(np.uint8)
    blended_img = Image.fromarray(blended_np)  # convert np array to PIL image

    # extract bounding boxes and draw with iou filtering

    boxes = []
    draw = ImageDraw.Draw(blended_img)

    # list of tuples for each labelled obj, each tuple has N=input_dim=2 slices,
    # each slice indicates smallest 3d parallelepiped containing obj. 3rd element of each slice is thus None for N=2
    slices = find_objects(labeled_mask)
    min_area_up = min_area * (scale_factor**2)
    for i, slc in enumerate(slices):
        if slc == None:
            continue

        # extract 2d box pixel coordinates
        y1, y2 = slc[0].start, slc[0].stop
        x1, x2 = slc[1].start, slc[1].stop

        # area filtering - skip annomaly if too small
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area_up:
            boxes.append([x1, y1, x2, y2, area])

    # sort boxes in descending area to prioritize larger areas
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    iou_thresh = 0.3
    kept_boxes = []

    # retain a box only if its IoU with all already-kept boxes is less than the threshold.
    for candidate in boxes:
        keep = True
        for kept in kept_boxes:
            if compute_iou(candidate, kept) > iou_thresh:
                keep = False
                break
        if keep:
            kept_boxes.append(candidate)

    for box in kept_boxes:
        x1, y1, x2, y2, _ = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    return blended_img


def annotate_frame(frame, type, font_size=14, stats=None):
    """Assumes frame is already PIL image"""

    draw = ImageDraw.Draw(frame)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    x, y = 5, 5
    if type == "data":
        draw.text((x, y), f"Data", fill=(255, 50, 50), font=font)
    elif type == "recon":
        draw.text((x, y), f"Reconstruction", fill=(255, 50, 50), font=font)
    elif type == "anomaly":
        draw.text((x, y), f"Anomaly Model", fill=(255, 50, 50), font=font)
    else:
        raise TypeError("Provided frame type not recognised for annotation")

    if stats:
        for key, value in stats.items():
            y += font_size + 2
            draw.text((x, y), f"{key}: {value:.3f}", fill=(255, 50, 50), font=font)

    return frame


def max_agg(recons, heatmaps, stride, clip_len, T, H, W):

    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )  # -> (T, H, W)

    final_recon = torch.zeros(
        (T, H, W), dtype=recons[0].dtype, device=recons[0].device
    )  # -> (T, H, W)

    for clip_idx, clip_err in enumerate(heatmaps):
        start_f = clip_idx * stride  # first frame covered by this clip
        recon_clip = recons[clip_idx]
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break

            mask = clip_err[local_t] > final_heat[global_f]
            final_recon[global_f][mask] = recon_clip[local_t][mask]

            final_heat[global_f] = torch.maximum(
                final_heat[global_f], clip_err[local_t]
            )  # -> (T, H, W)

            # alpha = 0.7  # or tuned hyperparam
            # final_heat[global_f] = (
            #     alpha * final_heat[global_f] + (1 - alpha) * clip_err[local_t]
            # )

    return final_heat, final_recon


def mean_agg(recons, heatmaps, stride, clip_len, T, H, W):

    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )  # -> (T, H, W)

    final_recon = torch.zeros(
        (T, H, W), dtype=recons[0].dtype, device=recons[0].device
    )  # -> (T, H, W)

    count_map = torch.zeros((T, 1, 1), dtype=torch.float32)

    for clip_idx, clip_err in enumerate(heatmaps):
        start_f = clip_idx * stride  # first frame covered by this clip
        recon_clip = recons[clip_idx]
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break
            final_heat[global_f] += clip_err[local_t]
            final_recon[global_f] += recon_clip[local_t]
            count_map[global_f] += 1

    # Avoid divide-by-zero
    count_map = count_map.clamp(min=1e-8)
    final_heat = final_heat / count_map
    final_recon = final_recon / count_map

    return final_heat, final_recon


def sum_agg(recons, heatmaps, stride, clip_len, T, H, W):

    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )  # -> (T, H, W)

    final_recon = torch.zeros(
        (T, H, W), dtype=recons[0].dtype, device=recons[0].device
    )  # -> (T, H, W)

    count_map = torch.zeros((T, 1, 1), dtype=torch.float32)

    for clip_idx, clip_err in enumerate(heatmaps):
        start_f = clip_idx * stride  # first frame covered by this clip
        recon_clip = recons[clip_idx]
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break
            final_heat[global_f] += clip_err[local_t]
            final_recon[global_f] += recon_clip[local_t]
            count_map[global_f] += 1

    # Avoid divide-by-zero
    count_map = count_map.clamp(min=1e-8)
    final_recon = final_recon / count_map

    return final_heat, final_recon


def percentile_agg(recons, heatmaps, stride, clip_len, T, H, W, percentile=90):
    """
    Aggregates overlapping frame-wise heatmaps and reconstructions using percentile-based aggregation.
    More robust than mean, less noisy than max.

    Args:
        recons: list of reconstruction clips (each of shape [clip_len, H, W])
        heatmaps: list of error maps (each of shape [clip_len, H, W])
        stride: frame stride between clips
        clip_len: number of frames per clip
        T, H, W: total time, height, width of the full video
        percentile: float in [0, 100], e.g., 90

    Returns:
        final_heat: Tensor of shape (T, H, W)
        final_recon: Tensor of shape (T, H, W), averaged recon
    """

    # Prepare aggregation containers
    final_recon = torch.zeros((T, H, W), dtype=recons[0].dtype, device=recons[0].device)
    recon_count = torch.zeros((T, 1, 1), dtype=torch.float32)

    # For percentile, we need to collect all overlapping heatmaps first
    heat_collector = [[] for _ in range(T)]

    for clip_idx, (clip_err, recon_clip) in enumerate(zip(heatmaps, recons)):
        start_f = clip_idx * stride
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break

            # Collect heatmap slices
            heat_collector[global_f].append(clip_err[local_t])

            # Still average recon frames normally
            final_recon[global_f] += recon_clip[local_t]
            recon_count[global_f] += 1

    # Compute percentile aggregation per frame
    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )
    for t in range(T):
        if heat_collector[t]:
            stacked = torch.stack(heat_collector[t], dim=0)  # shape: (N, H, W)
            final_heat[t] = torch.quantile(stacked, percentile / 100.0, dim=0)

    # Avoid divide-by-zero
    recon_count = recon_count.clamp(min=1e-8)
    final_recon = final_recon / recon_count

    return final_heat, final_recon


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
