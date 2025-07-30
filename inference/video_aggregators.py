# %% Imports

import torch

# %% Methods


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
