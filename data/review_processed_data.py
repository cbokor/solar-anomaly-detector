# %% Imports

import torch
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# %% Methods


def review_processed_data(save_stats=False):
    """Global script for full review pipeline
    - generate stats
    - plot stats
    - highlight clips beyond given thresh-hold
    - allow user to review each manually?
    - move chosen clips into anomaly file?
    """

    # Initialize var's
    root_dir = "data//processed"
    anomaly_dir = "data//processed//anomalies"
    os.makedirs(anomaly_dir, exist_ok=True)
    if save_stats:
        stats_csv = "//data//processed_clip_stats.csv"
    else:
        stats_csv = None

    # Generate stats directly
    all_stats = eval_all_clips_in_folder(root_dir, stats_csv)

    if not all_stats:
        print("[ERROR] No clips processed. Exiting,")
        return

    # Convert list of dict's to Dataframe
    data_frame = pd.DataFrame(all_stats)

    # Plot Distributions
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    data_frame["overall_mean"].hist(ax=ax[0], bins=30)
    ax[0].set_title("Overall Mean Distribution")
    data_frame["overall_std"].hist(ax=ax[1], bins=30)
    ax[1].set_title("Overall Std Distribution")
    plt.tight_layout()
    plt.show()

    # Extract a ref clip closest to the mean
    dataset_mean = data_frame["overall_mean"].mean()

    # Find index of closest clip to mean to act as ref:
    # gen array of difs between each clip and mean -> take abs value -> extract idx of smallest distance (min)
    ref_idx = (data_frame["overall_mean"] - dataset_mean).abs().idxmin()
    ref_clip_path = os.path.join(root_dir, data_frame.loc[ref_idx, "clip_path"])

    ref_clip = torch.load(ref_clip_path, map_location="cpu", weights_only=True)
    ref_T = ref_clip.shape[1]

    print(f"[INFO] Using {data_frame.loc[ref_idx, 'clip_path']} as reference clip.")

    # Flag potential outliers via nominated threshold

    # Threshold:
    # 95% rule: assuming Normal dist, 95% data falls within mean +- 2 std, upper 2.5% = anomaly.
    # options -> 1.645*std~90%; 1.44*std~85%, 1.28*std~80%, 1.15*std~75%, 1*std~68%
    cdf_z = 1.645
    mean_thresh = (
        data_frame["overall_mean"].mean() + cdf_z * data_frame["overall_mean"].std()
    )
    std_thresh = (
        data_frame["overall_std"].mean() + cdf_z * data_frame["overall_std"].std()
    )

    # Create a boolean mask of the stats data_frame,
    # generating True for a row if mean OR (|) std are above thresholds
    # outputs pd dataframe.shape(k,m) where k=no of flagged rows, m=no of stat columns
    flagged = data_frame[
        (data_frame["overall_mean"] > mean_thresh)
        | (data_frame["overall_std"] > std_thresh)
    ]

    print(f"[INFO] Flagged {len(flagged)} potential anomaly clips for review.")

    for idx, row in flagged.iterrows():
        clip_path = os.path.join(root_dir, row["clip_path"])
        print(f"\n[REVIEW] Clip: {row['clip_path']}")
        # `float:.3f` to specifiy float with 3 decimal places
        print(f"Stats: mean={row['overall_mean']:.3f}, std={row['overall_std']:.3f}")

        # load flagged clip
        clip = torch.load(clip_path, map_location="cpu", weights_only=True)
        C, T, H, W = clip.shape
        global_min = min(clip.min().item(), ref_clip.min().item())
        global_max = max(clip.max().item(), ref_clip.max().item())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
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

        # define local callback function for animation updates
        def update(frame_idx):
            # frame_idx % T → ensures that even if FuncAnimation loops beyond T frames,
            # it wraps back around (e.g., frame 17 with T=16 → 17%16=1).
            frame_flagged = clip[:, frame_idx % T, :, :].permute(1, 2, 0).squeeze()
            frame_ref = ref_clip[:, frame_idx % ref_T, :, :].permute(1, 2, 0).squeeze()

            im_flagged.set_data(frame_flagged)
            im_ref.set_data(frame_ref)  # update disp frame
            return [im_flagged, im_ref]

        # create FuncAnimation object: inputs: (
        # interval=100: time in ms between frames, i.e. 100ms = 10fps
        # blit=True: smoother animation by only updating parts of figure, can blur out per pixle var
        # )
        ani = animation.FuncAnimation(
            fig, update, frames=T, interval=100, blit=False, repeat=True
        )

        plt.show(block=False)  # open animation but dont block script

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
                    print("[INFO] Moved {clip_path} -> {dst}")
                break
            elif choice == "esc":
                plt.close(fig)
                return
            else:
                print("[INFO] Replaying clip...")
                plt.show(block=False)


def eval_clip_stats(clip: torch.Tensor):

    # Raise AssertionError with message if statement not True
    assert clip.ndim == 4, f"Expected shape [C T H W], got {clip.shape}"

    # Flattern 4D tensor to ease stats computation per channel.
    # Note flatten in this context litteraly means rearranging all elements into singuler vectors per channel via 'row-major order'
    # I.e., all pixels accross every frame, column, and row in the clip is assembled into a vector
    flattened = clip.view(
        clip.shape[0], -1
    )  # reshape 4D tensor: [C, T, H, W] -> [C, T*H*W]

    per_channel_mean = flattened.mean(
        dim=1
    ).tolist()  # output list of floats, 1 for each channel
    per_channel_std = flattened.std(dim=1).tolist()

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
    Recursively finds all .pt files under root_dir, evaluates statistics for each clip,
    and optionally saves the results to a CSV.

    Args:
        root_dir (str): Path to the root folder containing .pt clip files.
        stats_output_path (str, optional): Path to save a CSV of stats. If None, does not save.

    Returns:
        List[Dict]: List of stats dicts per clip.
    """

    all_stats = []

    # os.walk() outputs (recursively generates) several tuples for each dir within root_dir of form:
    # (current path, dir-names in current path: list, filenames in current path: list).
    # we ignor dir names via '_'
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pt"):
                clip_path = os.path.join(dirpath, fname)
                try:
                    # load specified file and perform operations
                    clip = torch.load(clip_path, map_location="cpu", weights_only=True)
                    stats = eval_clip_stats(clip)
                    stats["clip_path"] = os.path.relpath(clip_path, root_dir)

                    # append stats dict to global list
                    all_stats.append(stats)
                except Exception as e:
                    print(f"[ERROR] Failed in {clip_path}: {e}")

    if stats_output_path:
        import csv  # read/write csv's

        keys = (
            (all_stats[0].keys()) if all_stats else []
        )  # obtain column names (list) via dict keys in first stats dict
        with open(
            stats_output_path, "w", newline=""
        ) as f:  #  open/create output file for writing
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"[INFO] Saved stats .csv to {stats_output_path}")

    return all_stats
