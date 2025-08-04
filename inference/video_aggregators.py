# %% Imports

import torch

# %% Methods


def max_agg(recons, heatmaps, stride, clip_len, T, H, W):
    """
    Aggregates overlapping frame-wise heatmaps and reconstructions using max-based aggregation.
    Maximum visual impact, preserves strong activations, can exagerate noise or outlier peaks/trophs
    More highlight-focused than mean or sum — best when anomalies are rare and bright

    Args:
        recons: list of reconstruction clips (each of shape [clip_len, H, W])
        heatmaps: list of error maps (each of shape [clip_len, H, W])
        stride: frame stride between clips
        clip_len: number of frames per clip
        T, H, W: total time, height, width of the full video

    Returns:
        final_heat: Tensor of shape (T, H, W)
        final_recon: Tensor of shape (T, H, W), averaged recon
    """

    # Prepare aggregation containers on same device as inputs
    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )  # -> (T, H, W)

    final_recon = torch.zeros(
        (T, H, W), dtype=recons[0].dtype, device=recons[0].device
    )  # -> (T, H, W)

    # Iterate over all clips
    for clip_idx, clip_err in enumerate(heatmaps):
        start_f = clip_idx * stride  # Global frame index for clip start
        recon_clip = recons[clip_idx]
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break  # Don't overflow final video length

            # Determine where this clip's heatmap is stronger than previous max
            mask = clip_err[local_t] > final_heat[global_f]

            # Update the reconstruction selectively where this clip's heatmap is stronger
            final_recon[global_f][mask] = recon_clip[local_t][mask]

            # Update the max heatmap aggregation
            final_heat[global_f] = torch.maximum(
                final_heat[global_f], clip_err[local_t]
            )  # -> (T, H, W)

    return final_heat, final_recon


def mean_agg(recons, heatmaps, stride, clip_len, T, H, W):
    """
    Aggregates overlapping frame-wise heatmaps and reconstructions using mean-based aggregation.
    De-noises, smooth visual output, can blur high intentsity regions or hide sharp anomalies.
    More balanced than all others; less sensitive than max, less intense than sum.

    Args:
        recons: list of reconstruction clips (each of shape [clip_len, H, W])
        heatmaps: list of error maps (each of shape [clip_len, H, W])
        stride: frame stride between clips
        clip_len: number of frames per clip
        T, H, W: total time, height, width of the full video

    Returns:
        final_heat: Tensor of shape (T, H, W)
        final_recon: Tensor of shape (T, H, W), averaged recon
    """

    # Prepare aggregation containers on same device as inputs
    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )  # -> (T, H, W)

    final_recon = torch.zeros(
        (T, H, W), dtype=recons[0].dtype, device=recons[0].device
    )  # -> (T, H, W)

    # Initialize frame counter map
    count_map = torch.zeros((T, 1, 1), dtype=torch.float32)

    # Accumulate sums and count contributions per frame
    for clip_idx, clip_err in enumerate(heatmaps):
        start_f = clip_idx * stride  # Global frame index for clip start
        recon_clip = recons[clip_idx]
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break  # Don't overflow final video length

            final_heat[global_f] += clip_err[local_t]
            final_recon[global_f] += recon_clip[local_t]
            count_map[global_f] += 1

    # Normalize to compute the mean, avoid divide-by-zero
    count_map = count_map.clamp(min=1e-8)
    final_heat = final_heat / count_map
    final_recon = final_recon / count_map

    return final_heat, final_recon


def sum_agg(recons, heatmaps, stride, clip_len, T, H, W):
    """
    Aggregates overlapping frame-wise heatmaps and reconstructions using sum-based aggregation.
    Preserves total energy, shows freqauncy or intentisty of total activations, but requires normalization and can oversaturate.
    Like a boosted mean; use only if values are scaled or you want to emphasize repetition.

    Args:
        recons: list of reconstruction clips (each of shape [clip_len, H, W])
        heatmaps: list of error maps (each of shape [clip_len, H, W])
        stride: frame stride between clips
        clip_len: number of frames per clip
        T, H, W: total time, height, width of the full video

    Returns:
        final_heat: Tensor of shape (T, H, W)
        final_recon: Tensor of shape (T, H, W), averaged recon

    Notes:
        only heatmap used sum_agg, recon is still a mean aggregation.
    """

    # Prepare aggregation containers on same device as inputs
    final_heat = torch.zeros(
        (T, H, W), dtype=heatmaps[0].dtype, device=heatmaps[0].device
    )  # -> (T, H, W)

    final_recon = torch.zeros(
        (T, H, W), dtype=recons[0].dtype, device=recons[0].device
    )  # -> (T, H, W)

    # Initialize frame counter map (only for recon, not heatmap)
    count_map = torch.zeros((T, 1, 1), dtype=torch.float32)

    for clip_idx, clip_err in enumerate(heatmaps):
        start_f = clip_idx * stride  # Global frame index for clip start
        recon_clip = recons[clip_idx]
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break  # Don't overflow final video length

            final_heat[global_f] += clip_err[local_t]
            final_recon[global_f] += recon_clip[local_t]
            count_map[global_f] += 1

    # Normalize to compute the mean (for recon only), avoid divide-by-zero
    count_map = count_map.clamp(min=1e-8)
    final_recon = final_recon / count_map

    return final_heat, final_recon


def percentile_agg(recons, heatmaps, stride, clip_len, T, H, W, percentile=99):
    """
    Aggregates overlapping frame-wise heatmaps and reconstructions using percentile-based aggregation.
    Felxable filtering, reduce outlier impact, highlight consistant strong signals.
    Computationally more expensive and picking right percentile is sensitive.
    A middle-ground between mean and max; gives control over noise vs signal tradeoff.

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
        start_f = clip_idx * stride  # Global frame index for clip start
        for local_t in range(clip_len):
            global_f = start_f + local_t
            if global_f >= T:
                break  # Don't overflow final video length

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
