# CUDA MVS

Another attempt at **Multi-View Stereo (MVS)** — this time using CUDA.

## Input Data

The program uses input data generated with [RealityScan](https://www.realityscan.com/):

- OpenCV-compatible camera parameters (`.csv`)
- Undistorted images (`.png`)
- Sparse point cloud (`.xyz`)

## Output

Generates **dense-ish point clouds**, one for each reference camera.

## Dependencies

- OpenCV 4.8
- CUDA 12.4
