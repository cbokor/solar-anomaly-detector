# %% Goal

# All data specific functions in this module must do the following for a particular data source (using data agnostic methods):
# - extract raw, heavy, astronomy-oriented data (.tar)→
# - Unpack it into a temporary directory →
# - Filter for only relevant files (e.g., .fits) →
# - Process and clean it →
# - Save to compact model-friendly files ready for training with PyTorch!

# -> next steps must be done elsewhere:
# + turn clips into batches and thus a DataLoader (proabably in train.py via a custom Dataset class)

# %% Imports

import os
import tarfile
import numpy as np
import torch
from PIL import Image
from astropy.io import fits
from tempfile import TemporaryDirectory
from tqdm import tqdm


# %% Methods (data agnostic)

# Notes for development:
# - print & tqdm.write commands currently placeholders for creating a logger
# - currently wasting anything not divisible by clip_len (default 16),


def extract_tar_to_temp(temp_path, tar_path, allowed_ext=None):
    """
    Extracts all files from a tar archive to a temporary directory.
    Args:
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

        for member in tqdm(
            members, desc=f"[INFO] Extracting", unit="file", leave=False
        ):

            try:
                # Log list of files being extracted and wether they are viable windows filenames
                invalid_chars = r'<>:"/\|?*'
                base_name = os.path.basename(member.name)
                if any(char in base_name for char in invalid_chars):
                    safe_name = sanitize_filename(
                        member.name
                    )  # Sanitize filename if it contains invalid characters
                    # tqdm.write(
                    #    f"    [INFO] Extracted {safe_name}: Contained invalid characters for Windows."
                    # )
                else:
                    safe_name = base_name
                    # tqdm.write(f"    [INFO] Extracted: {member.name}")

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

    # Collect all .fits files from the temporary directory (note: assumes filenames
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

    return matching_files, temp_obj  # tempdir must be kept open by the caller


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
    Reads a FITS file, normalizes the image data, resizes it, and returns it as a numpy array.

    Note on FITS files (Flexible Image Transport System):
    - FITS is a digital file format used for storing astronomical data.
    - It is widely used in astronomy for storing images, spectra, and other scientific data or metadata.
    - FITS files can contain multiple HDU's (Header/Data Units), each with its own header and data.
    - HDU list obj example: hdul[0], hdul[1], ...
    - the primary HDU (hdul[0]) usually contains the main image data, but sometimes the data is in a secondary HDU (hdul[1]).

    Example HDU in a fits file:
    ```
    >>> hdul.info()
    Filename: aia.lev1.304A_2023-11-01T00_00_05.13Z.image_lev1.fits
    No.    Name         Type      Cards   Dimensions   Format
    0    PRIMARY     PrimaryHDU     178   ()           float32
    1                ImageHDU       112   (4096, 4096) float32
    ```

    """

    hdul = fits.open(fits_path)  # create header data unit list object from fits file
    img = (
        hdul[1].data if len(hdul) > 1 else hdul[0].data
    )  # Get the image data from the first HDU or the primary HDU
    hdul.close()

    img = np.nan_to_num(img)  # Handle any NaNs
    img = (img - np.min(img)) / (
        np.max(img) - np.min(img) + 1e-8
    )  # Normalize to [0,1] range (avoiding division by zero)

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

    return np.array(img)  # Return as specified precision numpy array


# %% Methods (data specific)


def prepare_solar_data(tar_dir, out_dir, config):
    """
    Prepares solar data for training by processing .fits files stored in .tar files or the specified dir.
    """

    # Import all local config variables
    clip_len = config["data_pre_processing"]["clip_length"]
    resize = tuple(config["data_pre_processing"]["resize"])
    precision = config["data_pre_processing"]["precision"]
    # if stride=clip_len, then fully sequential clips, ideally stride=1,2,or4 for sliding window
    stride = config["data_pre_processing"]["stride"]

    os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists
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
            # but can be manually cleaned up to ensure no temp files are left
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
            )  # keep temp_obj for cleanup later

        elif entry.endswith(".fits"):
            all_fits_files.append((entry_path, None))  # single file, no temp_obj

        else:  # skip if not a .tar or .fits
            tqdm.write(f"\n[INFO] Skipping unsupported file: {entry}")
            continue

    if not all_fits_files:
        print("[ERROR] No .fits files found in directory or tar archives")
        return

    # Process and assemble all frames

    frames = []
    for entry_name, fits_list in tqdm(
        all_fits_files,
        desc="[INFO] Assembling all frames",
        unit="frame_packs",
        position=0,
        leave=False,
    ):

        if not fits_list:
            try:
                frame = process_fits_image(
                    entry_name, resize=resize, precision=precision
                )
                frames.append(frame)
            except Exception as e:
                tqdm.write(f"   [ERROR] Failed to process {entry_name}: {e}")
                continue
        else:
            for fits_file in tqdm(
                fits_list,
                desc=f"[INFO] Pre-processing .fits files from archive",
                unit="file",
                position=1,  # place above outer bar
                leave=False,  # do not leave this bar on screen after finishing
            ):

                try:
                    frame = process_fits_image(
                        fits_file, resize=resize, precision=precision
                    )
                    frames.append(frame)
                except Exception as e:
                    tqdm.write(f"   [ERROR] Failed to process {fits_file}: {e}")
                    continue

    if len(frames) < clip_len:
        tqdm.write(
            f"  [WARNING] Not enough frames in {frames} for a full clip of length {clip_len}. Skipping..."
        )
        return

    if len(frames) % clip_len != 0:

        lost = len(frames) % clip_len
        tqdm.write(
            f"  [WARNING] Number of total frames is not divisible by clip length {clip_len}. "
        )
        tqdm.write(f"  Truncating to the nearest multiple: ({lost}) frames were lost).")

        frames = frames[: len(frames) - lost]

    frames = np.stack(
        frames
    )  # Stack frames into a numpy array (shape: [num_frames, height, width, channels])

    # check for accepted precision types
    if precision == "uint8":
        data_type = torch.uint8
    elif precision == "float32":
        data_type = torch.float32
    else:
        raise ValueError("Unsupported precision type. Use 'uint8' or 'float32'.")

    # Create one full movie tensor of data for later evaluation
    movie_tensor = torch.tensor(frames, dtype=data_type).unsqueeze(
        1
    )  # (T, H, W) -> (T, C=1, H, W)
    # convert to expected format for CNN's in Pyorch,
    # forward compatable to support multi-channel input in the future (e.g., 171A & 304A)
    movie_tensor = movie_tensor.permute(1, 0, 2, 3)  # -> (C=1, T, H, W)
    movie_dir = os.path.join(out_dir, "full_movie_train")
    os.makedirs(movie_dir, exist_ok=True)
    out_name = "full_movie.pt"
    out_path = os.path.join(movie_dir, out_name)
    torch.save(movie_tensor, out_path)

    # Create clips of specified length for training + validation
    clips = [
        frames[i : i + clip_len] for i in range(0, len(frames) - clip_len + 1, stride)
    ]

    for i, clip in tqdm(
        enumerate(clips),
        desc=f"[INFO] Saving clips:",
        unit="clip",
    ):

        # Convert to tensor and add channel dimension (assuming grey scale images)
        clip_tensor = torch.tensor(clip, dtype=data_type).unsqueeze(
            1
        )  # (T, H, W) -> (T, C=1, H, W)

        # convert to expected format for CNN's in Pyorch,
        # forward compatable to support multi-channel input in the future (e.g., 171A & 304A)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # -> (C=1, T, H, W)

        out_name = f"clip_{i:04d}.pt"
        out_path = os.path.join(out_dir, out_name)
        torch.save(clip_tensor, out_path)

    # Clean up all temporary directories
    for _, temp_obj in all_fits_files:
        if temp_obj is not None:
            temp_obj.cleanup()
