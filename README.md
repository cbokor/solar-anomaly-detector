# 3D convolutional Auto Encoder for Anomaly Detection

Unsupervised learning for computer vision based anomaly detection via a modular PyTorch pipeline and reconstructive 3D-AE.

**Not much time to read?** See the sample gif below (first 4s) or [watch a full demo video!](demo/anomaly_video_low_level.mp4)

![Demo gif](demo/demo.gif)

## Key Features

- 📦 Modular PyTorch pipeline (training, preprocessing, evaluation)
- 🧠 Interchangeable architecture, example based on Mengjia et al., (2020) 3D convolutional autoencoder
- 📸 Applied to NASA SDO solar data — complex, nonlinear spatiotemporal signals
- 📊 Easily configurable via `config.yaml` and an args parser for reproducible testing and video output
- 🖥️ Includes demo videos and command-line usage
- 🔧 Easily adapted for alternative applications (surveillance, industrial fault detection, robotics, etc.).

## Overview 

This repository contains a modular PyTorch pipeline for testing and evaluating various computer vision-based anomaly detection architectures, including reconstructive, predictive, and generative models. These tasks are critical in domains such as:

- Surveillance and security  
- Industrial fault monitoring  
- Defence and autonomous systems  
- Space weather and astrophysics 

To demonstrate, I provide a reproducible implementation of [Mengjia et al. (2020)](http://dx.doi.org/10.1016/j.jvcir.2019.102747), applying a 3D convolutional autoencoder to detect spatiotemporal anomalies in time-series imagery of the Sun (e.g., solar flares, coronal mass ejections).

> While this project emphasizes modularity and embedded-friendly design, it is **not** currently optimized for real-time or resource-constrained deployment.

> Example settings (e.g., 112×112 image resolution, batch size, worker count) are chosen to reflect local GPU limitations. These can be adapted via `config.yaml`. A sample [`config.yaml`](config.yaml) is provided.

---

## Example Results - "Mother's Day Solar Storm" 2024

Solar imagery provides an excellent real-world testbed for spatiotemporal anomaly detection. Instruments like **AIA** on the **Solar Dynamics Observatory (SDO)** observe the Sun in multiple extreme ultraviolet (EUV) and UV wavelengths (e.g., **171 Å**, **304 Å**, **1700 Å**), each sensitive to different plasma temperatures and solar layers (corona, transition region, photosphere). This leads to complex signal characteristics and dynamics such as:

- Quasi-cyclic behavior with low-amplitude fluctuations,
- Continuously evolving baselines due to solar rotation and structural change,
- Strong nonlinear and sometimes chaotic temporal dynamics,
- High spatial heterogeneity across active regions and quiet Sun,
- A wide range of transient phenomena, from short impulsive bursts to long-duration flares.

These properties make solar video data well-suited for benchmarking **unsupervised, generalizable anomaly detection** systems in both space and time.

To demonstrate the capabilities of this repository, I analyzed data from the **“Mother’s Day Solar Storms”** in **May 2024**. As reported by [Mara Johnson-Groh (NASA, 2024)](https://science.nasa.gov/science-research/heliophysics/how-nasa-tracked-the-most-intense-solar-storm-in-decades/):

> “During the first full week of May, a barrage of large solar flares and coronal mass ejections (CMEs) launched clouds of charged particles and magnetic fields toward Earth, creating the strongest solar storm to reach Earth in two decades — and possibly one of the strongest displays of auroras on record in the past 500 years.”

For evaluation, I used AIA **171 Å** data from **00:00 UTC May 10 to 00:00 UTC May 12, 2024**, sampled at **1-minute cadence**. These dates include a sequence of large flares and CMEs described in NASA’s coverage for [May 10](https://science.nasa.gov/blogs/solar-cycle-25/2024/05/10/strong-solar-flare-erupts-from-sun-16/) and [May 11](https://science.nasa.gov/blogs/solar-cycle-25/2024/05/11/sun-releases-2-strong-flares-2/).

The imagery was acquired using the [SunPy](https://sunpy.org) Python library and the [Virtual Solar Observatory (VSO)](https://sdac.virtualsolar.org/cgi/search). All data were resized to **112×112 pixels** due to GPU constraints, as specified in `config.yaml`.

### 🎥 Demo Videos

The `demo/` folder includes two videos showcasing anomaly detection results using the 3D convolutional autoencoder based on [Mengjia et al. (2020)](http://dx.doi.org/10.1016/j.jvcir.2019.102747):

**Legend:**  
> **Red Boxes** = low-level anomalies (2% ≤ reconstruction loss < 8%)
>
> **Light Blue Boxes** = high-level anomalies (reconstruction loss ≥ 8%)
>
> **Top left Number of Boundary Boxes** = live reconstruction loss (0->1, i.e., 0.08 = 8%)
>
> **Global Stats** = the mean, max and standard deviation reconstruction loss values for the entire frame

These thresholds are defined in `config.yaml` as `threshold_low` and `threshold_high`. Training was done with the provided example [`config.yaml`](config.yaml).

- **(a) [Low-Level Dynamics Video](demo/anomaly_video_low_level.mp4)**  
  Left panel emphasizes subtle activity and regular solar intensity fluctuations; right panel contains residual heatmap and anomaly bounding boxes.

- **(b) [High-Intensity Events Video](demo/anomaly_video_high_level.mp4)**  
  Left panel colourscale shifted to focus on intense solar events (e.g., major flares); right panel again shows the same residual heatmap and anomaly bounding boxes.

All major activity was successfully captured using a **moderately deep 3D autoencoder** (13 parameterized layers, no skip connections, wide early layers). While training is GPU-intensive, the final model is suitable for **inference in embedded systems** after post-processing (e.g., quantization, pruning).

---

## Training Data

The dataset used during development is **not included** (task-specific/proprietary). However, the pipeline:

- Accepts `.fits` and `.tar` files  
- Includes built-in resizing (default: 112×112 px)  
- Can be easily adapted to other domains (e.g., surveillance, autonomous vehicles)  

To replicate results using solar data, I recommend:

- Using [SunPy](https://sunpy.org) to access AIA/SDO imagery via VSO  
- Selecting **quiet solar days** for training  
- Carefully tuning `stride` and `sample_interval_sec` in `config.yaml` to avoid overfitting  

See `prepare_solar_data()` in [`data/prepare_data.py`](data/prepare_data.py) for integration details.

---

## Usage

The repo includes four separate pipeline modes all run from `main.py` that must be selected at the command line via the `-m` or `--mode` arg. All other args are optional or have a default value (in `main.py` see `parse_args()` for more information). The modes include: 

+ `"prep"` -> tools for processing raw .fits and .tar files into Torch tensors suitable for training. Runs [`prepare_solar_data()`](data/prepare_data.py) from `data/prepare_data.py`. Always generates a video of provided raw files. Has an arg mode to generate only the video and not clips for later evaluation. 

+ `"review"` -> automatic and manual tools for reviewing processed data clips (e.g, manually select clips with anomalies in and segregate from training data, use intensity stats to automatically segregate outlier clips). Runs [`review_processed_data()`](data/review_processed_data.py) from `data/review_processed_data.py`.

+ `"train"` -> full training pipeline using training clips with summary writer logging. Optional tensorboard visual monitoring and prior training checkpoint loading provided. Runs [`train_model()`](training/train.py) from `training/train.py`.

+ `"eval"` -> evaluate a given torch tensor movie via a trained model to output a .mp4 movie. Several optional features are included (e.g., diagnostic or standard output video, boundary box inclusion or just heatmap, etc). Runs [`evaluate_model()`](inference/evaluate.py) from `inference/evaluate`.

For further information on any of the specific features included in these modes please see the thorough documentation for each respectively. 

Minimal example operation then:
```bash
python main.py -m "train"
```

Alternatively, one can call:
```bash
python main.py
```
and use an override `sys.argv` block in main.py as demonstrated for use in debugging/testing: 
```bash
if __name__ == "__main__":

    # Can override args for testing/debugging
    sys.argv = [
        "main.py",
        "--mode",
        "eval",
        "--config",
        "config.yaml",
        "--tensorboard",
        "--movie_only",
    ]

    main()
```
---

## Requirements

A full requirements file of the environment used during construction is provided but the lightweight summary is:

- Python 3.11
- PyTorch (>=2.0)
- torchvision
- numpy
- matplotlib
- opencv-python
- sunpy
- astropy
- scipy
- pandas
- PyYAML
- tqdm
- tensorboard

## Acknowledgments

A big thank you to:
- The creators of [Detecting Spatiotemporal Irregularities in Videos via a 3D Conv AE (2020)](http://dx.doi.org/10.1016/j.jvcir.2019.102747).
- The [PyTorch](https://pytorch.org/) community for the core deep learning enviroment.
- The [SunPy Project](https://sunpy.org) for open-source solar data analysis tools, including support for querying and downloading data via VSO, and structured solar metadata (Fido, Time, Instrument, etc.).
- [NASA/Solar Data Analysis Center (SDAC)](https://umbra.nascom.nasa.gov/index.html/) and the [VSO](https://sdac.virtualsolar.org/cgi/search) teams for hosting and maintaining distributed solar data repositories accessible through the VSO API.

## License & Citation

This project is licensed under the [MIT License](./LICENSE).

You are free to use, modify, and distribute this code.  
If you find it useful in your work, please consider citing it using the metadata in [`CITATION.cff`](./CITATION.cff).

## Recommended Reading

Below are some key papers I'm fond of in the realm of anomaly detection:

- [Detecting Spatiotemporal Irregularities in Videos via a 3D Conv AE (2019)](http://dx.doi.org/10.1016/j.jvcir.2019.102747)
- [ASTNet: Attention Residual Autoencoder for Video Anomaly Detection (2022)](https://link.springer.com/article/10.1007/s10489-022-03613-1) or the [GitHub_Repo](https://github.com/vt-le/astnet?tab=readme-ov-file#readme)
- [SMAMS: Spatiotemporal Masked Autoencoder with Multi-Memory (2024)](https://doi.org/10.3390/electronics13020353)