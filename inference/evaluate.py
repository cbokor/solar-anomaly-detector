# %% Imports

import os
import torch
import yaml
import numpy as np
import imageio.v2 as imageio
import matplotlib.cm as cm
import models.conv3d_autoencoder as conv3d_autoencoder
from utils.image_utils import to_rgb, upscale_array, upscale_image, percent_norm
from inference.video_aggregators import percentile_agg
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, find_objects
from training.loss_functions import LOSS_REGISTRY

# %% Methods


def evaluate_model(args):
    """
    Evaluate a 3D autoencoder model on a full-length video clip.
    Outputs either a standard anomaly video or full diagnostic video based on args.

    Assumptions:
        - Input video exists in specified folder and is shaped (1, T, H, W)

    Evaluation Workflow:
        1. Load the trained model and configuration.
        2. Load the full video to be evaluated.
        3. Slide a window across the video to extract clips of length T=clip_len.
        4. Run the model on each clip to produce reconstructions.
        5. Compute pixel-wise reconstruction errors (i.e., anomaly heatmaps).
        6. Aggregate overlapping heatmaps into a single frame-wise heatmap.
        7. Optionally normalize, annotate, and overlay heatmaps or bounding boxes.
        8. Concatenate visual results and save the final annotated video.

    Outputs:
        - Annotated .mp4 file containing side-by-side visual diagnostics.

    Raises:
        - ValueError: If no recognizable eval mode provided.
    """

    # Initialize file paths
    movie_dir = os.path.join(args.data_clips, "full_movie_eval", "full_movie.pt")
    output_path = os.path.join(args.data_clips, "full_movie_eval", "anomaly_video.mp4")
    model_dir = os.path.join(args.model_path, "best_model", "best_model.pth.tar")
    config_path = os.path.join(args.model_path, "config.yaml")

    # Load config file into local var as hierarchal dict
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Read evaluation settings
    clip_len = config["data_pre_processing"]["clip_length"]
    stride = config["evaluate"]["stride"]
    threshold = config["evaluate"]["threshold"]
    min_area = config["evaluate"]["min_area"]
    weight = config["evaluate"]["heat_weight"]
    scale_factor = config["evaluate"]["scale_factor"]
    fps = config["evaluate"]["fps"]
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

    # Load and shape input video: (1, T, H, W)
    video = torch.load(movie_dir)  # shape: (T, H, W) or (1, T, H, W)

    if video.ndim == 3:
        video = video.unsqueeze(0)  # ensure shape is (1, T, H, W)

    C, T, H, W = video.shape

    # Generate sliding clips of video
    clips = []
    for i in range(0, T - clip_len + 1, stride):
        clip = video[:, i : i + clip_len, :, :]
        clips.append(clip)

    # Run inference and collect reconstructions/errors
    heatmaps = []  # shape (T, H, W)
    recons = []  # shape (T, H, W)
    for clip in tqdm(clips, desc=f"Reconstructing Clips"):
        input_clip = clip.unsqueeze(0).to(device)  # shape: (1, 1, T, H, W)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=torch.cuda.is_available()
        ):
            recon = model(input_clip)

        # Generate error maps for each frame in clip
        error = torch.abs(recon - input_clip).squeeze(0).squeeze(0)  # shape (T, H, W)

        recons.append(recon.squeeze(0).squeeze(0).cpu())
        heatmaps.append(error.cpu())

    # Aggregate overlapping heatmaps (e.g., via percentile-based aggregation)
    # Several options available: max_agg(), mean_agg(), sum_agg(), percentile_agg()
    final_heat, final_recon = percentile_agg(
        recons, heatmaps, stride, clip_len, T, H, W
    )

    # Convert tensors to numpy arrays
    video_np = video.squeeze(0).numpy()  # (T, H, W)
    recon_np = final_recon.cpu().numpy()  # (T, H, W)
    heat_np = final_heat.cpu().numpy()  # (T, H, W)

    # Setup video writer for heat_map/boxes overlay
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

    # Main rendering loop
    if args.eval_diagnostic == False:

        # Render eval video for anomaly detection only
        for t in tqdm(range(T), desc="Assembling Eval .mp4"):

            frame = video_np[t]
            heat = heat_np[t]
            recon = recon_np[t]

            # Normalization per frame based on data, not recon
            frame, recon = percent_norm(frame, recon)

            # Apply colour maps
            cmap = cm.get_cmap(
                "magma"
            )  # -> cmap(x) return ([R, G, B, A]) for x in [0,1]
            frame = cmap(frame)[
                :, :, :3
            ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)

            # Ensure frame is uint8 RGB
            frame_rgb = to_rgb(frame)

            # Generate upscaled heatmap with overlay info
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
                "Frame": t,
                "Mean": heat.mean(),
                "Max": heat.max(),
                "Std": heat.std(),
            }

            # Annotate frames
            frame_pil = annotate_frame(frame_pil, "data")
            anomaly_pil = annotate_frame(anomaly_pil, "anomaly", stats=stats)

            # Concatenate horizontally
            row = Image.new("RGB", (frame_pil.width * 2, frame_pil.height))
            row.paste(frame_pil, (0, 0))
            row.paste(anomaly_pil, (frame_pil.width, 0))

            writer.append_data(np.array(row))

        print(f"[INFO] Anomaly video saved to {output_path}. Mode = {args.eval_mode}")

    else:

        # Render eval video with full diagnostic output

        for t in tqdm(range(T), desc="Assembling Eval .mp4"):
            frame = video_np[t]
            heat = heat_np[t]
            recon = recon_np[t]

            # Normalization per frame based on data, not recon
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

            # Ensure frames are uint8 RGB
            frame_rgb = to_rgb(frame)
            recon_rgb = to_rgb(recon)

            # Generate upscaled heatmap with overlay info
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

        print(
            f"[INFO] Diagnostic anomaly video saved to {output_path}. Mode = {args.eval_mode}"
        )

    writer.close()


def overlay_heatmap(frame_grey, heat_map, weight, scale_factor):
    """
    Overlay a heatmap onto a grayscale frame using a colormap and alpha blending.

    Args:
        frame_grey (np.ndarray): Grayscale input frame; shape (H, W), dtype float,
                                 in range [0, 1].
        heat_map (np.ndarray): Anomaly heatmap; shape (H, W), dtype float, expected in [0, 1].
        weight (float): Blend weight for heatmap overlay; 0.0 = only frame, 1.0 = only heatmap.
        scale_factor (float): Scaling factor for the final upscaled output image.

    Returns:
        PIL.Image: An upscaled RGB image showing the grayscale frame blended with a colormapped heatmap.

    Notes:
        - Percentile normalization is applied to the heatmap for contrast enhancement.
        - `frame_grey` is converted to RGB and assumed to be properly normalized.
        - Output image is suitable for visual inspection, not precise metric computation.
    """

    # Normalize heatmap for better visual contrast
    heat_map = percent_norm(heat_map)

    # Optional: z_normed per frame (isnt great for visual but good for enhanced post-processing)
    # mean = heat_map.mean()
    # std = heat_map.std()
    # heat_map = (heat_map - mean) / (std + 1e-8)

    # Convert grayscale frame to RGB uint8 format
    frame_rgb = to_rgb(frame_grey)  # (H,W)->(H,W,3)

    # Apply colormap to heatmap (e.g., "jet") and remove alpha channel
    cmap = cm.get_cmap("jet")  # -> cmap(x) return ([R, G, B, A]) for x in [0,1]
    heat_colored = cmap(heat_map)[
        :, :, :3
    ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)
    heat_colored = to_rgb(heat_colored)

    # Blend grayscale frame and color heatmap with the specified weight
    blended_np = ((1 - weight) * frame_rgb + weight * heat_colored).astype(np.uint8)

    # Convert np array to PIL image
    blended_img = Image.fromarray(blended_np)

    # Upscale result and return
    return upscale_image(blended_img, scale_factor)


def overlay_heatmap_with_boxes(
    frame_grey, heat_map, threshold, min_area, weight, scale_factor
):
    """
    Overlay anomaly heatmap and draw bounding boxes on top of a grayscale frame.

    This method includes:
        - Percentile-based normalization of the heatmap
        - Thresholding and area-based region filtering
        - Non-Maximum Suppression (NMS) using IoU with area prioritization
        - Blending of the original frame with a colored heatmap
        - Drawing red bounding boxes for detected anomaly regions

    Args:
        frame_grey (np.ndarray): Grayscale image, shape (H, W), dtype float.
        heat_map (np.ndarray): Heatmap of anomaly scores, shape (H, W), in [0,1].
        threshold (float): Pixel-level threshold to create binary anomaly mask (post-normalization).
        min_area (int): Minimum pixel area to consider a region anomalous (pre-scale).
        weight (float): Blend weight between original frame and heatmap [0,1].
        scale_factor (float): Upscaling factor applied before drawing boxes.

    Returns:
        PIL.Image: RGB image with heatmap overlay and bounding boxes.
    """

    # Upscale input arrays for higher-resolution visualization
    frame_grey_up = upscale_array(frame_grey)
    heat_map_up = upscale_array(heat_map)

    # Convert grayscale frame to RGB uint8
    frame_rgb = to_rgb(frame_grey_up)  # (H,W)->(H,W,3)

    # Create binary anomaly mask using threshold, (H,W)
    mask = (heat_map_up > threshold).astype(np.uint8)

    # Label connected components in the binary mask (background = 0), (H, W)
    labeled_mask, num_labels = label(mask)

    # Normalize heatmap for better visual contrast (specifically AFTER anomaly mask in original scale)
    heat_map_error_scale = heat_map_up
    heat_map_up = percent_norm(heat_map_up)

    # Apply colormap to heatmap and remove alpha channel
    cmap = cm.get_cmap("jet")
    heat_colored = cmap(heat_map_up)[
        :, :, :3
    ]  # provide map (H,W,4) -> i.e., [R G B A] per pixel; remove alpha 'A' for (H,W,3)
    heat_colored = to_rgb(heat_colored)

    # Blend grayscale RGB frame with heatmap
    blended_np = ((1 - weight) * frame_rgb + weight * heat_colored).astype(np.uint8)
    blended_img = Image.fromarray(blended_np)  # convert np array to PIL image

    # Prepare for drawing bounding boxes
    boxes = []
    draw = ImageDraw.Draw(blended_img)

    # Find bounding boxes for labeled regions
    # list of tuples for each labelled obj, each tuple has N=input_dim=2 slices,
    # each slice indicates smallest 3d parallelepiped containing obj. 3rd element of each slice is thus None for N=2
    slices = find_objects(labeled_mask)
    min_area_up = min_area * (scale_factor**2)

    for i, slc in enumerate(slices):
        if slc == None:
            continue

        # Extract 2d box pixel coordinates
        y1, y2 = slc[0].start, slc[0].stop
        x1, x2 = slc[1].start, slc[1].stop

        # Apply area filtering - skip annomaly if too small
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area_up:
            boxes.append([x1, y1, x2, y2, area])

    # Sort boxes in descending area to prioritize larger areas
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    iou_thresh = 0.2
    kept_boxes = []

    # Apply Non maximum suppresion, I.e.,
    # retain a box only if its IoU with all already-kept boxes is less than the threshold.
    for candidate in boxes:
        keep = True
        for kept in kept_boxes:
            if compute_iou(candidate, kept) > iou_thresh:
                keep = False
                break
        if keep:
            kept_boxes.append(candidate)

    # Calculate anomaly score for each box (i.e., mean heatmap value inside the box)
    scored_boxes = []
    for box in kept_boxes:
        x1, y1, x2, y2, area = box
        patch = heat_map_error_scale[y1:y2, x1:x2]
        score = float(np.max(patch))
        scored_boxes.append([x1, y1, x2, y2, score])

    # Sort by score descending
    scored_boxes = sorted(scored_boxes, key=lambda b: b[4], reverse=True)

    # Set up colors: red for others, green/orange/yellow for top 3
    high_color = (0, 255, 255)  # 2.5x higher than threshold: green
    default_color = (255, 0, 0)  # normal: red

    # Optional font setup
    font_size = 8
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        font2 = ImageFont.truetype("arial.ttf", font_size + 2)
    except:
        font = ImageFont.load_default()
        font2 = ImageFont.load_default()

    for i, box in enumerate(scored_boxes):
        x1, y1, x2, y2, score = box
        color = high_color if box[4] > (4 * threshold) else default_color

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw score text just above box
        color2 = high_color if box[4] > (4 * threshold) else (255, 255, 255)
        text = f"{score:.2f}"
        draw.text((x1 + 2, y1 - font_size - 2 - 1), text, fill=color2, font=font2)

    return blended_img


def annotate_frame(frame, type, font_size=12, stats=None):
    """
    Annotate a given PIL image with a label and optional per-frame statistics.

    Args:
        frame (PIL.Image): The input image to annotate (assumed to be in RGB).
        type (str): Type of frame, used to determine label.
                    Options: "data", "recon", "anomaly".
        font_size (int): Font size for annotations.
        stats (dict, optional): Dictionary of per-frame stats to display.
                                Keys will be shown with float values (3 decimals).

    Returns:
        PIL.Image: The annotated image.

    Raises:
        TypeError: If an unrecognized frame type is provided.
    """

    draw = ImageDraw.Draw(frame)

    # Load preferred font, fallback to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Starting position for text
    width, height = frame.size
    label_color = (255, 50, 50)
    x, y = 5, 5

    # text_width, _ = draw.textsize(url_text, font=font)

    # Draw the frame label based on the type with underline
    if type == "data":
        title = f"Data"
        draw.text(
            (x, height - font_size - 5),
            "github.com/cbokor",
            fill=label_color,
            font=font,
        )
    elif type == "recon":
        title = f"Reconstruction"
    elif type == "anomaly":
        title = f"Anomaly Model"
    else:
        raise TypeError("Provided frame type not recognised for annotation")

    bbox = draw.textbbox((x, y), title, font=font)
    text_width = bbox[2] - bbox[0]
    # text_height = bbox[3] - bbox[1]
    draw.text((x, y), title, fill=label_color, font=font)
    underline_y = y + font_size + 2
    draw.line((x, underline_y, x + text_width, underline_y), fill=label_color, width=1)

    # Optionally draw statistics below the label
    if stats:
        title = "Global Stats:"
        bbox = draw.textbbox((x, y), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = width - text_width - 5
        draw.text((x, y), title, fill=label_color, font=font)
        for key, value in stats.items():
            y += font_size + 2
            if key == "Frame":
                draw.text((x, y), f"{key}: {value:.0f}", fill=label_color, font=font)
            else:
                draw.text((x, y), f"{key}: {value:.3f}", fill=label_color, font=font)

    return frame


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list or tuple): Bounding box in the format [x1, y1, x2, y2].
        boxB (list or tuple): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0  # No overlap

    # Compute area of both bounding boxes
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union and IoU
    union_area = float(areaA + areaB - inter_area)
    iou = inter_area / union_area

    return iou
