# %% Imports

import os
import tarfile
import numpy as np
import torch
import re
from PIL import Image
from astropy.io import fits
from tempfile import TemporaryDirectory
from tqdm import tqdm
from datetime import datetime


# %% Methods (data agnostic)


def fname_to_datetime(fname):
    """
    Extracts a datetime object from AIA or JSOC FITS filenames based on embedded timestamps.

    This function supports the following filename timestamp formats:
    1. Old AIA format with underscores:
       e.g., 'aia.lev1.171A_2012_04_10T00_05_01.35Z.image_lev1.fits'
       Corresponding pattern: YYYY_MM_DDTHH_MM_SS

    2. New JSOC format with hyphens:
       e.g., 'jsoc-aia.lev1.304A_2023-11-01T00-00-05.13Z.image_lev1.fits'
       Corresponding pattern: YYYY-MM-DDTHH-MM-SS

    Parameters:
        fname (str): The filename containing the timestamp.

    Returns:
        datetime: The extracted timestamp as a `datetime` object.

    Raises:
        ValueError: If no recognizable timestamp format is found in the filename.
    """

    # Get the base name of the file (remove directory path)
    basename = os.path.basename(fname)

    # Try to match the old format: e.g., 2012_04_10T00_05_01
    match_old = re.search(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})", basename)
    if match_old:
        return datetime.strptime(match_old.group(1), "%Y_%m_%dT%H_%M_%S")

    # Try to match the new format: e.g., 2023-11-01T00-00-05
    match_new = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", basename)
    if match_new:
        return datetime.strptime(match_new.group(1), "%Y-%m-%dT%H-%M-%S")

    # Raise an error if neither format matched
    raise ValueError(f"Could not extract timestamp from filename: {basename}")


def extract_tar_to_temp(temp_path, tar_path, allowed_ext=None):
    """
    Extracts all files from a tar archive to a temporary directory.
    Args:
        temp_path (str): Location to store temp dir's
        tar_path (str): Path to the tar file.
        allowed_ext (list, optional): List of allowed file extensions to filter files.
                                      If None, all files are included. Defaults to None.

    Returns:
        matching_files (list of str): List of full paths to the extracted files that match the allowed extensions.
        temp_obj (TemporaryDirectory): Temporary directory object containing the extracted files.

    """
    temp_obj = TemporaryDirectory(dir=temp_path)  # Create a temporary directory object
    temp_dir = temp_obj.name  # Get the name of the temporary directory

    with tarfile.open(
        tar_path
    ) as tar:  # Open the tar file temporarily, ensure it is closed after extraction

        members = [m for m in tar.getmembers() if not m.isdir()]
        success_count = 0
        fail_count = 0

        # Loop through files in tar archive and extract to temp dir. Provide count of failures and successes.
        # Remove tqdm bar after completion
        for member in tqdm(
            members, desc=f"[INFO] Extracting", unit="file", leave=False
        ):

            try:
                # Check wether member names are viable windows filenames, alter accordingly
                invalid_chars = r'<>:"/\|?*'
                base_name = os.path.basename(member.name)
                if any(char in base_name for char in invalid_chars):
                    safe_name = sanitize_filename(
                        member.name
                    )  # Sanitize filename if it contains invalid characters
                else:
                    safe_name = base_name

                # Extract file object and write with safe_name
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    out_path = os.path.join(temp_dir, safe_name)
                    with open(out_path, "wb") as f:
                        f.write(extracted_file.read())
                    success_count += 1

            except Exception as e:
                tqdm.write(f"    [ERROR] Failed to extract {member.name}: {e}")
                fail_count += 1

        tqdm.write(
            f"[INFO] Extraction Summary: {success_count} files extracted, {fail_count} failed."
        )

    # Collect all .fits files from the temp dir (note: assumes filenames
    # will naturally order frames chronologically)
    all_files = [
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if os.path.isfile(os.path.join(temp_dir, f))
    ]

    if allowed_ext:
        # Filter files by allowed extensions if specified (split into root, ext; only check ext as [1])
        # Assumes allowed_ext is a list of lower case strings
        matching_files = [
            f for f in all_files if os.path.splitext(f)[1].lower() in allowed_ext
        ]
    else:
        matching_files = all_files

    return (
        matching_files,
        temp_obj,
    )  # tempdir must be kept open by the caller and closed elsewhere


def sanitize_filename(filename):
    """
    Sanitizes a filename by removing invalid characters for Windows filenames.
    Args:
        filename (str): The original filename.

    Returns:
        str: The sanitized filename with invalid characters replaced by underscores.
    """
    invalid_chars = r'<>:"/\|?*'
    for char in filename:
        if char in invalid_chars:
            filename = filename.replace(char, "-")  # Replace invalid chars with hyphen
    return filename  # Return the sanitized filename


def process_fits_image(fits_path, resize=(112, 112), precision="float32"):
    """
    Loads a astronomical FITS file, normalizes the image data, resizes it, and returns it as a numpy array.

    The function performs the following steps:
    1. Opens the FITS file using `astropy.io.fits`.
    2. Extracts image data from the most likely HDU (Header/Data Unit).
    3. Handles NaNs and normalizes pixel values to [0, 1].
    4. Resizes the image to the given shape using PIL.
    5. Converts to specified floating-point precision.
    6. Returns the processed image as a NumPy array.

    Note on FITS files (Flexible Image Transport System):
    - FITS is a digital file format used for storing astronomical data.
    - It is widely used in astronomy for storing images, spectra, and other scientific data or metadata.
    - FITS files can contain multiple HDU's (Header/Data Units), each with its own header and data.
    - HDU list obj example: hdul[0], hdul[1], ...
    - the primary HDU (hdul[0]) usually contains the main image data, but sometimes the data is in a secondary HDU (hdul[1]).

    Parameters:
        fits_path (str): Path to the input FITS file.
        resize (tuple): Output resolution as (width, height). Default is (112, 112).
        precision (str): Output data precision, must be either 'float16' or 'float32'.

    Returns:
        np.ndarray: The resized and normalized image as a NumPy array with the specified precision.

    Raises:
        ValueError: If an unsupported precision type is given.

    Example HDU in a fits file:
        >>> hdul.info()
        Filename: aia.lev1.304A_2023-11-01T00_00_05.13Z.image_lev1.fits
        No.    Name         Type      Cards   Dimensions   Format
        0    PRIMARY     PrimaryHDU     178   ()           float32
        1                ImageHDU       112   (4096, 4096) float32

    """

    hdul = fits.open(fits_path)  # Create header data unit list object from fits file

    # Use the second HDU if available, otherwise fall back to the primary HDU
    img = hdul[1].data if len(hdul) > 1 else hdul[0].data

    hdul.close()  # Close the FITS file to free memory

    # Replace any NaN values with zero to avoid issues during processing
    img = np.nan_to_num(img)

    # Normalize to [0,1] range (avoiding division by zero)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    # Resize and cast the image to the desired precision
    if precision == "float16":
        # Convert to Lower precision PIL Image and resize (16-bit format, [0,1])
        img = Image.fromarray(img.astype(np.float16)).resize(resize)

    elif precision == "float32":
        # Convert to Higher Precision PIL Image and resize (32-bit float format, [0,1])
        img = Image.fromarray(img.astype(np.float32)).resize(resize)
    else:
        raise ValueError(
            "Unsupported precision type for training. Use 'float16' or 'float32'."
        )

    return np.array(img)  # Return numpy array


# %% Methods (data specific)


def prepare_solar_data(tar_dir, out_dir, config, movie_only=False):
    """
    Prepares solar image data from FITS files (direct or inside .tar archives) into normalized, resized, and timestamped
    PyTorch tensors suitable for training video-based models.

    This function supports a bespoke pre-processing pipeline tailored for solar observation data. It:
    - Recursively extracts `.fits` files from the specified directory and any `.tar` archives within it.
    - Converts FITS images to normalized grayscale frames using configured precision and resolution.
    - Sorts frames by timestamp (parsed from filenames) and detects any data gaps based on time discontinuities.
    - Assembles all frames into a single "movie" tensor and saves it.
    - Splits the sequence into overlapping clips of fixed length with optional stride, saving them separately.
    - Clips containing data gaps are saved to a separate subdirectory for downstream filtering or evaluation.

    Parameters:
        tar_dir (str): Path to the input directory containing `.fits` files or `.tar` archives with `.fits` files inside.
        out_dir (str): Path to the output directory where processed tensors and clips will be saved.
        config (dict): A dictionary containing pre-processing parameters, structured as:
            {
                "data_pre_processing": {
                    "clip_length" (int): Number of frames per training clip.
                    "resize" (list/tuple): Target image resolution as (width, height), e.g. (112, 112).
                    "precision" (str): Data precision for output tensors, either "float16" or "float32".
                    "stride" (int): Number of frames to shift between each generated clip.
                    "sample_interval_sec" (int): Expected time (in seconds) between consecutive frames.
                    "gap_threshhold" (int or float): Threshold multiplier for gap detection; gaps are flagged if time between frames > sample_interval_sec * gap_threshhold.
                }
            }

    Outputs:
        - A `full_movie.pt` tensor saved to `<out_dir>/full_movie_train/`, shaped (C=1, T, H, W).
        - Individual training clips saved to `<out_dir>/` or `<out_dir>/gapped_clips/` depending on presence of time gaps.
          Each clip is saved as `clip_XXXX.pt` with shape (C=1, T, H, W).

    Raises:
        ValueError: If the specified precision is not "float16" or "float32".
        Exception: If insufficient valid frames are found to create a single clip.

    Notes:
        - Time gaps are detected based on parsed timestamps in filenames; clips with such gaps are isolated for robustness.
        - Temporary directories created during archive extraction are cleaned up after processing.
        - FITS image data is assumed to be grayscale (1-channel) for simplicity, but the code is forward-compatible with multi-channel data.
    """

    # Import all local config variables
    clip_len = config["data_pre_processing"]["clip_length"]
    resize = tuple(config["data_pre_processing"]["resize"])
    precision = config["data_pre_processing"]["precision"]
    stride = config["data_pre_processing"]["stride"]
    cadence_sec = config["data_pre_processing"]["sample_interval_sec"]
    gap_threshhold = config["data_pre_processing"]["gap_threshhold"]

    os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists

    # Ensure dir for clips with identified gaps exists
    gapped_dir = os.path.join(out_dir, "gapped_clips")
    os.makedirs(gapped_dir, exist_ok=True)

    all_fits_files = []

    # Cycle through all .tar and .fits files in given dir
    for entry in tqdm(
        sorted(os.listdir(tar_dir)),
        desc="[INFO] Processing .tar Archives & .fits Files",
        unit="archive/file",
        position=0,  # pin outer bar to the bottom
        leave=True,  # leave it on screen after finishing
    ):

        entry_path = os.path.join(tar_dir, entry)

        if entry.endswith(".tar"):
            tqdm.write(f"\n[INFO] Processing Archive: {entry}")
            # Extract all .fits files from the tar archive to a temporary directory
            # Note: temp_obj will be deleted automatically when it goes out of scope
            # but is manually cleaned up at end of function to ensure no temp files are left
            fits_files, temp_obj = extract_tar_to_temp(
                out_dir, entry_path, allowed_ext=[".fits"]
            )
            if not fits_files:
                tqdm.write(
                    f"    [WARNING] No .fits files found in {entry}. Skipping..."
                )
                continue
            all_fits_files.append(
                (fits_files, temp_obj)
            )  # Store extracted archive files with temp_obj for cleanup later

        elif entry.endswith(".fits"):
            all_fits_files.append(
                (entry_path, None)
            )  # Store single file, no temp_obj needed

        else:  # Skip if not a .tar or .fits
            tqdm.write(f"\n[INFO] Skipping unsupported file: {entry}")
            continue

    if not all_fits_files:
        print("[ERROR] No .fits files found in directory or tar archives")
        return

    # Process and assemble all frames from gathered fits files
    frames = []
    for fits_list, temp_obj in tqdm(
        all_fits_files,
        desc="[INFO] Assembling all frames",
        unit="frame_packs",
        position=0,
        leave=False,
    ):

        # Check if entry is a single file or a group of files from an archive
        if isinstance(fits_list, str):
            try:
                frame = process_fits_image(
                    fits_list, resize=resize, precision=precision
                )
                timestamp = fname_to_datetime(fits_list)
                frames.append((timestamp, frame))
            except Exception as e:
                tqdm.write(f"   [ERROR] Failed to process {fits_list}: {e}")
                continue
        else:
            for fits_file in tqdm(
                fits_list,
                desc=f"[INFO] Pre-processing .fits files from archive",
                unit="file",
                position=1,  # place above outer bar
                leave=False,  # do not leave this bar on screen after finishing
            ):

                # Process frame from each fits file in archive bundle
                try:
                    frame = process_fits_image(
                        fits_file, resize=resize, precision=precision
                    )
                    timestamp = fname_to_datetime(fits_file)
                    frames.append((timestamp, frame))
                except Exception as e:
                    tqdm.write(f"   [ERROR] Failed to process {fits_file}: {e}")
                    continue

    # Check if insufficent frames were extracted to form a clip
    if len(frames) < clip_len:
        raise Exception(
            "[ERROR] Not enough frames in provided dir for a full clip of specified length {clip_len}"
        )

    # Check if any frames remain after maximum number of full clips are made via specified clip_length
    if len(frames) % clip_len != 0:

        lost = len(frames) % clip_len
        tqdm.write(
            f"  [WARNING] Number of total frames is not divisible by clip length {clip_len}. "
        )
        tqdm.write(f"  Truncating to the nearest multiple: ({lost}) frames were lost).")

        frames = frames[: len(frames) - lost]

    # Sort frames by timestamp
    frames.sort(key=lambda x: x[0])
    timestamps = [t for t, _ in frames]

    # Identify any gaps in timestamps of collective frames and store as a bool mask
    # flag gaps > gap_threshhold x cadence
    gap_flags = [False] * len(timestamps)
    for i in range(1, len(timestamps)):
        if (
            timestamps[i] - timestamps[i - 1]
        ).total_seconds() > gap_threshhold * cadence_sec:
            gap_flags[i] = True  # Mark first frame after a gap

    frames = np.stack(
        [f[1] for f in frames]
    )  # Stack frames into a numpy array (shape: [num_frames, height, width, channels])

    # Check for accepted precision types
    if precision == "float16":
        data_type = torch.float16
    elif precision == "float32":
        data_type = torch.float32
    else:
        raise ValueError(
            "Unsupported precision type for training. Use 'float16' or 'float32'."
        )

    # Create one full movie tensor of data for later evaluation
    movie_tensor = torch.tensor(frames, dtype=data_type).unsqueeze(
        1
    )  # (T, H, W) -> (T, C=1, H, W)

    # Convert tensor channel order to expected format for CNN's in Pyorch,
    # forward compatable to support multi-channel input in the future (e.g., 171A & 304A)
    movie_tensor = movie_tensor.permute(1, 0, 2, 3)  # -> (C=1, T, H, W)

    # Create dir for movie and save
    movie_dir = os.path.join(out_dir, "full_movie_train")
    os.makedirs(movie_dir, exist_ok=True)
    out_name = "full_movie.pt"
    out_path = os.path.join(movie_dir, out_name)
    torch.save(movie_tensor, out_path)

    if movie_only == False:
        # Create clips of specified length for training + validation
        for i, start in tqdm(
            enumerate(range(0, len(frames) - clip_len + 1, stride)),
            desc=f"[INFO] Saving clips:",
            unit="clip",
        ):
            end = start + clip_len
            clip = frames[start:end]

            # If any gap flag inside this clip is True, mark to be send to gapped_dir
            has_gap = any(
                gap_flags[start + 1 : end]
            )  # +1 so a gap marks the *start* of missing interval

            # Convert to tensor and add channel dimension (assuming grey scale images)
            clip_tensor = torch.tensor(clip, dtype=data_type).unsqueeze(
                1
            )  # (T, H, W) -> (T, C=1, H, W)

            # Convert to expected format for CNN's in Pyorch,
            # forward compatable to support multi-channel input in the future (e.g., 171A & 304A)
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # -> (C=1, T, H, W)

            # Assign dir accordingly per has_gap and save
            subdir = gapped_dir if has_gap else out_dir
            out_name = f"clip_{i:04d}.pt"
            out_path = os.path.join(subdir, out_name)
            torch.save(clip_tensor, out_path)

    # Clean up all temporary directories
    for _, temp_obj in all_fits_files:
        if temp_obj is not None:
            temp_obj.cleanup()
