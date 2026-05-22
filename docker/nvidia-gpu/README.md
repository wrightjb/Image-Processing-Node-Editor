# Dockerfile

Base environment

- Python3.8
- Docker (Tested on NVIDIA-Docker2)
- Webcam (connected to /dev/video0)
- Desktop PC (Intel CPU + NVIDIA RTX2080Ti)

<br>

## Build

```bash
git clone https://github.com/Kazuhito00/Image-Processing-Node-Editor.git
cd Image-Processing-Node-Editor/
docker build docker/nvidia-gpu -t ipn_editor
```

<br>

## Run

The following command worked. If you stop the container, caches are removed, so mount any external folder as needed.

```bash
# cd /path-to-Image-Processing-Node-Editor
xhost +
docker run --rm -it --privileged --device /dev/video0:/dev/video0:mwr -e DISPLAY=$DISPLAY -v $(pwd):/workspace --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix ipn_editor
# A window opens automatically
```

### About the options used

- `--device /dev/video0:/dev/video0:mwr`: Remove this if you do not connect a webcam.
- `--gpus all`: Do not use this option on non-NVIDIA GPUs.
- `-v $(pwd):/workspace`: [Image-Processing-Node-Editor](https://github.com/Kazuhito00/Image-Processing-Node-Editor) mount destination.
  - These options assume execution from inside the `Image-Processing-Node-Editor` directory.
  - The mount target is `/workspace`.