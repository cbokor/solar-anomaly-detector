# %% Imports

import torch
import os
import shutil
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# %% Methods


def review_processed_data(root_dir, save_stats=False):
    """
    Interactive pipeline to review preprocessed solar data clips for statistical anomalies.

    This function performs the following:
    - Computes summary statistics (mean, std) for all `.pt` clips in a given directory.
    - Plots histograms of the mean and standard deviation across all clips.
    - Flags outliers using a configurable z-score threshold (default: z=1.645 ~95% confidence).
    - Allows the user to visually compare flagged clips to a "reference" (mean) clip using an animation.
    - Prompts the user to classify each flagged clip as anomalous (`y`) or not (`n`).
    - Moves accepted anomalies to a designated `anomalies/` subfolder.

    Parameters:
        root_dir (str): Directory containing `.pt` clips to review (e.g., "data/processed").
        save_stats (bool): If True, saves computed statistics as a CSV file in `root_dir/stats`.

    Outputs:
        - Displays histogram plots of mean and std values.
        - Displays animated side-by-side comparisons of flagged and reference clips.
        - Moves clips confirmed as anomalies to `root_dir/anomalies/`.

    Raises:
        None explicitly, but prints errors if files are missing or corrupt.
    """

    # Initialize var's

    # Override root_dir for consistency
    root_dir = "data//processed"

    # Define path for storing accepted anomalies
    anomaly_dir = os.path.join(root_dir, "anomalies")
    os.makedirs(anomaly_dir, exist_ok=True)

    # Optional CSV output path
    stats_csv = os.path.join(root_dir, "stats") if save_stats else None

    # Compute summary statistics for all clips
    all_stats = eval_all_clips_in_folder(root_dir, stats_csv)

    if not all_stats:
        print("[ERROR] No clips processed. Exiting,")
        return

    # Convert stats (list of dict's) to Dataframe
    data_frame = pd.DataFrame(all_stats)

    # Plot Distributions of per-clip statistics
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    data_frame["overall_mean"].hist(ax=ax[0], bins=30)
    ax[0].set_title("Overall Mean Distribution")
    data_frame["overall_std"].hist(ax=ax[1], bins=30)
    ax[1].set_title("Overall Std Distribution")
    plt.tight_layout()
    plt.show()

    # Locate a ref clip - closest to dataset mean
    # Method: gen array of difs between each clip and mean -> take abs value -> extract idx of smallest distance (min)
    dataset_mean = data_frame["overall_mean"].mean()
    ref_idx = (data_frame["overall_mean"] - dataset_mean).abs().idxmin()
    ref_clip_path = os.path.join(root_dir, data_frame.loc[ref_idx, "clip_path"])

    # Load ref clip
    ref_clip = torch.load(ref_clip_path, map_location="cpu", weights_only=True)
    ref_T = ref_clip.shape[1]

    print(f"[INFO] Using {data_frame.loc[ref_idx, 'clip_path']} as reference clip.")

    # Set anomaly threshold: mean + cdf_z * std
    # 95% rule: assuming Normal dist, 95% data falls within mean +- 2 std, upper 2.5% = anomaly.
    # Other options -> 1.645*std~90%; 1.44*std~85%, 1.28*std~80%, 1.15*std~75%, 1*std~68%
    cdf_z = 1.645
    mean_thresh = (
        data_frame["overall_mean"].mean() + cdf_z * data_frame["overall_mean"].std()
    )
    std_thresh = (
        data_frame["overall_std"].mean() + cdf_z * data_frame["overall_std"].std()
    )

    # Identify clips that exceed thresholds via a boolean mask
    # outputs pd dataframe.shape(k,m) where k=no of flagged rows, m=no of stat columns
    flagged = data_frame[
        (data_frame["overall_mean"] > mean_thresh)
        | (data_frame["overall_std"] > std_thresh)
    ]

    print(f"[INFO] Flagged {len(flagged)} potential anomaly clips for review.")

    # Review each flagged clip interactively
    for idx, row in flagged.iterrows():
        clip_path = os.path.join(root_dir, row["clip_path"])
        print(f"\n[REVIEW] Clip: {row['clip_path']}")
        # `float:.3f` to specifiy float with 3 decimal places
        print(f"Stats: mean={row['overall_mean']:.3f}, std={row['overall_std']:.3f}")

        # Load flagged clip
        clip = torch.load(clip_path, map_location="cpu", weights_only=True)
        C, T, H, W = clip.shape

        # Global min/max to sync visualization scaling
        global_min = min(clip.min().item(), ref_clip.min().item())
        global_max = max(clip.max().item(), ref_clip.max().item())

        # Static side-by-side frame preview
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Convert first frame for display
        # permute = [C, H, W] -> [H, W, C] for matplotlib
        # squeeze() = remove any dimensions of size 1: [C=1, H, W] -> [H, W]
        frame_img_flagged = clip[:, 0, :, :].permute(1, 2, 0).squeeze()
        im_flagged = ax[0].imshow(
            frame_img_flagged,
            cmap="grey" if C == 1 else None,
            vmin=global_min,
            vmax=global_max,
        )
        ax[0].set_title(f"Flagged Clip")
        ax[0].axis("off")

        frame_img_ref = ref_clip[:, 0, :, :].permute(1, 2, 0).squeeze()
        im_ref = ax[1].imshow(
            frame_img_ref,
            cmap="grey" if C == 1 else None,
            vmin=global_min,
            vmax=global_max,
        )
        ax[1].set_title(f"Reference Clip")
        ax[1].axis("off")

        # Local animation update function
        def update(frame_idx):
            # frame_idx % T → ensures that even if FuncAnimation loops beyond T frames,
            # it wraps back around (e.g., frame 17 with T=16 → 17%16=1).
            frame_flagged = clip[:, frame_idx % T, :, :].permute(1, 2, 0).squeeze()
            frame_ref = ref_clip[:, frame_idx % ref_T, :, :].permute(1, 2, 0).squeeze()

            im_flagged.set_data(frame_flagged)
            im_ref.set_data(frame_ref)  # update disp frame
            return [im_flagged, im_ref]

        # Create FuncAnimation object, animating both clips side by side:
        # inputs: (interval=100: time in ms between frames, i.e. 100ms = 10fps
        # blit=True: smoother animation by only updating parts of figure, can blur out per pixle var
        # )
        ani = animation.FuncAnimation(
            fig, update, frames=T, interval=100, blit=False, repeat=True
        )

        plt.show(block=False)  # open animation but dont block script

        # Prompt user for classification
        while True:
            choice = (
                input(
                    (
                        "[INPUT] Press [enter] to watch again; 'y'/'n' to classify;"
                        "or type 'esc' to end review."
                    )
                )
                .strip()
                .lower()
            )
            if choice in ("y", "n"):
                plt.close(fig)
                if choice == "y":
                    dst = os.path.join(anomaly_dir, os.path.basename(clip_path))
                    shutil.move(clip_path, dst)
                    print(f"[INFO] Moved {clip_path} -> {dst}")
                break
            elif choice == "esc":
                plt.close(fig)
                return
            else:
                print("[INFO] Replaying clip...")
                plt.show(block=False)


def eval_clip_stats(clip: torch.Tensor):
    """
    Compute basic statistical metrics for a 4D video clip tensor.

    Parameters:
        clip (torch.Tensor): A 4D tensor with shape [C, T, H, W] representing
                             channels, time (frames), height, and width.

    Returns:
        dict: A dictionary containing:
            - shape (list[int]): Original shape of the clip.
            - per_channel_mean (list[float]): Mean pixel value per channel.
            - per_channel_std (list[float]): Standard deviation per channel.
            - overall_mean (float): Mean pixel value across entire clip.
            - overall_std (float): Std deviation across entire clip.
    """

    # Validate tensor has expected 4D shape
    assert clip.ndim == 4, f"Expected shape [C T H W], got {clip.shape}"

    # Flatten spatial and temporal dimensions: [C, T, H, W] -> [C, T*H*W]
    # I.e., This arranges all pixels across time and space into a single vector per channel
    flattened = clip.view(clip.shape[0], -1)

    # Compute mean and std per channel
    per_channel_mean = flattened.mean(
        dim=1
    ).tolist()  # output list of floats, 1 for each channel
    per_channel_std = flattened.std(dim=1).tolist()

    # Compute overall mean and std across all values
    overall_mean = (
        flattened.mean().item()
    )  # calc mean -> output 1 element tensor -> item() convernt to float
    overall_std = flattened.std().item()  # as above but std

    stats = {
        "shape": list(clip.shape),
        "per_channel_mean": per_channel_mean,
        "per_channel_std": per_channel_std,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
    }

    return stats


def eval_all_clips_in_folder(root_dir, stats_output_path=None):
    """
    Recursively evaluates statistical summaries for all .pt video clip tensors in a directory.

    This function:
    - Iterates through all .pt files in the given directory (non-recursive).
    - Loads each tensor using torch.load().
    - Computes statistical summaries using `eval_clip_stats()`.
    - Optionally writes the statistics for all clips to a CSV file.

    Args:
        root_dir (str): Path to the folder containing `.pt` clip files (each a [C, T, H, W] tensor).
        stats_output_path (str, optional): If provided, saves statistics to a CSV at this location.

    Returns:
        List[Dict]: A list of statistics dictionaries, one per clip.
    """

    all_stats = []

    for fname in os.listdir(root_dir):
        clip_path = os.path.join(root_dir, fname)

        # Only process regular .pt files
        if os.path.isfile(clip_path) and fname.endswith(".pt"):
            try:
                # Load the tensor; ensure it is on CPU
                clip = torch.load(clip_path, map_location="cpu", weights_only=True)

                # Evaluate statistics
                stats = eval_clip_stats(clip)

                # Store filename for identification
                stats["clip_path"] = fname

                all_stats.append(stats)
            except Exception as e:
                print(f"[ERROR] Failed in {clip_path}: {e}")

    # If an output path is specified, save all stats to a CSV
    if stats_output_path:

        # Ensure output dir exists
        os.makedirs(stats_output_path, exist_ok=True)

        # Compose full path
        stats_output_path = os.path.join(stats_output_path, "all_clips_stats.csv")

        # Obtain column names (list) via dict keys in first stats dict
        keys = (all_stats[0].keys()) if all_stats else []

        # Write stats to csv
        with open(
            stats_output_path, "w", newline=""
        ) as f:  #  open/create output file for writing
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_stats)

        print(f"[INFO] Saved stats .csv to {stats_output_path}")

    return all_stats
