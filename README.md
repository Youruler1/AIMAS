# AIMAS
Consists of a simple and easy-to-use voice conversion framework based on VITS.

---

> The base model is trained on nearly 50 hours of high-quality open-source VCTK dataset. No copyright concerns, feel free to use it.

> Stay tuned for the RVCv3 base model: larger parameters, more data, better performance, nearly the same inference speed, and requiring less training data.

---

## Introduction
This repository has the following features:
+ Uses top1 retrieval to replace input source features with training set features to prevent timbre leakage
+ Fast training even on relatively weak GPUs
+ Produces good results with only a small amount of data (recommended at least 10 minutes of clean speech data)
+ Model fusion can be used to change timbre (via the ckpt-merge option in the ckpt tab)
+ Simple and user-friendly web interface
+ Can call UVR5 models to quickly separate vocals and accompaniment
+ Uses the state-of-the-art [vocal pitch extraction algorithm InterSpeech2023-RMVPE](#references) to eliminate mute problems. Significantly better results, faster than crepe_full, and lighter in resource usage
+ Supports acceleration for AMD and Intel GPUs

Check out our [demo video](https://www.bilibili.com/video/BV1pm4y1z7Gm/)!

---

## Environment Setup
Commands must be run in Python 3.8 or higher.

### Windows/Linux/MacOS (General Methods)
Choose one of the following methods:

#### 1. Install dependencies via pip
1. Install Pytorch and dependencies (skip if already installed). Reference: https://pytorch.org/get-started/locally/
```bash
pip install torch torchvision torchaudio
```
2. If on Windows + Nvidia Ampere (RTX30xx), install with the matching CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
3. Install dependencies based on your GPU:
- Nvidia (CUDA):
```bash
pip install -r requirements.txt
```
- AMD/Intel:
```bash
pip install -r requirements-dml.txt
```
- AMD ROCm (Linux):
```bash
pip install -r requirements-amd.txt
```
- Intel IPEX (Linux):
```bash
pip install -r requirements-ipex.txt
```

#### 2. Install dependencies via Poetry
Install Poetry (skip if already installed): https://python-poetry.org/docs/#installation
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
When using Poetry, Python 3.7–3.10 is recommended (newer versions may conflict with llvmlite==0.39.0).
```bash
poetry init -n
poetry env use "path to your python.exe"
poetry run pip install -r requirements.txt
```

### MacOS
Run the script:
```bash
sh ./run.sh
```

---

## Additional Pretrained Models
RVC requires some pretrained models for inference and training.

Download them from our [Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

### 1. Download assets
List of required models/files (scripts available in `tools/` folder):
- ./assets/hubert/hubert_base.pt
- ./assets/pretrained
- ./assets/uvr5_weights

For v2 models, also download:
- ./assets/pretrained_v2

### 2. Install ffmpeg
If already installed, skip.

Ubuntu/Debian:
```bash
sudo apt install ffmpeg
```
MacOS:
```bash
brew install ffmpeg
```
Windows: place downloaded files in the root directory:
- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. Download RMVPE vocal pitch extraction files
Download [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) and place in root directory.

(Optional for AMD/Intel GPUs) Download [rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)

### 4. AMD GPU ROCm (Linux only)
Install ROCm drivers: https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html

For Arch Linux:
```bash
pacman -S rocm-hip-sdk rocm-opencl-sdk
```

Extra configuration for some GPUs (e.g., RX6700XT):
```bash
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```
Ensure user is in `render` and `video` groups:
```bash
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
```

---

## Getting Started

### Start directly
```bash
python infer-web.py
```
With Poetry:
```bash
poetry run python infer-web.py
```

### Using bundled package
Download and extract `RVC-beta.7z`.

Windows:
- Double-click `go-web.bat`

MacOS:
```bash
sh ./run.sh
```

### Intel IPEX (Linux only)
```bash
source /opt/intel/oneapi/setvars.sh
```

---

## References
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [RMVPE Pitch Extraction](https://github.com/Dream-High/RMVPE)

---

## Thanks
Thanks to all contributors:  
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
