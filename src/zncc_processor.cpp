#include "../include/zncc_processor.h"
#include "../include/cuda_homography.h"
#include <cuda_runtime.h>
#include <stdio.h>

ZNCCProcessor::ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius)
    : imgWidth(imgWidth), imgHeight(imgHeight), windowRadius(windowRadius) {
    setWindowRadius(windowRadius);
    znccOutput.gpuData = nullptr;
    znccOutput.width = imgWidth;
    znccOutput.height = imgHeight;
    znccOutput.pitch = 0;
}

ZNCCProcessor::~ZNCCProcessor() {
    if (znccOutput.gpuData) {
        cudaFree(znccOutput.gpuData);
    }
}

void ZNCCProcessor::setCameraMatrices(const float* H0, const float* K1inv, const float* e2) {
    setupCameraMatrices(H0, K1inv, e2);
}

ImageData ZNCCProcessor::computeZNCC(const ImageData& refImage, const ImageData& targetImage,
    const ImageData& depthMap, const ImageData& normalMap) {
    // Allocate output with pitch for consistency
    if (!znccOutput.gpuData) {
        cudaMallocPitch(&znccOutput.gpuData, &znccOutput.pitch, imgWidth * sizeof(float), imgHeight);
    }    

    // Call wrapper function with separate dimensions for ref and target images
    launchEvaluateZNCC(refImage.gpuData, targetImage.gpuData, depthMap.gpuData, normalMap.gpuData,
        refImage.width, refImage.height, targetImage.width, targetImage.height,
        znccOutput.gpuData,
        refImage.pitch, targetImage.pitch, depthMap.pitch, normalMap.pitch,
        config.blockDim.x, config.blockDim.y, config.gridDim.x, config.gridDim.y,
        znccOutput.pitch);

    checkCudaError("computeZNCC kernel launch");

    return znccOutput;
}