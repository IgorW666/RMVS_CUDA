#include "../include/image_handler.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdexcept>

ImageHandler::~ImageHandler() {
    // Cleanup handled by individual image free calls
}
ImageData ImageHandler::loadImageToGPU(const std::string& filename) {
    // Load image with OpenCV (grayscale)
    cv::Mat cpuImage = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (cpuImage.empty()) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    if (false) {    
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(cpuImage, cpuImage);
    }

    printf("DEBUG: Loaded image dimensions: %d x %d\n", cpuImage.cols, cpuImage.rows);
    printf("DEBUG: Image step: %zu bytes, %zu floats per row\n",
        (size_t)cpuImage.step[0], (size_t)cpuImage.step[0] / sizeof(float));

    // Convert to float32 normalized [0, 1]
    cv::Mat floatImage;
    cpuImage.convertTo(floatImage, CV_32F, 1.0f / 255.0f);

    printf("DEBUG: After convertTo - dims: %d x %d, step: %zu bytes, %zu floats per row\n",
        floatImage.cols, floatImage.rows, (size_t)floatImage.step[0],
        (size_t)floatImage.step[0] / sizeof(float));

    // Allocate GPU memory
    ImageData result = allocateGPUImage(floatImage.cols, floatImage.rows);

    printf("DEBUG: GPU allocation - GPU pitch: %zu bytes, %zu floats per row\n",
        result.pitch, result.pitch / sizeof(float));

    // Copy to GPU with proper pitch handling
    copyHostToGPU(reinterpret_cast<float*>(floatImage.data), result, floatImage.step[0]);

    return result;
}

ImageData ImageHandler::loadImageToGPU(const cv::Mat& mat)
{
    if (mat.empty()) {
        throw std::runtime_error("Input cv::Mat is empty");
    }

    // Normalize only 8-bit input; keep float data (depth/normals) unchanged
    cv::Mat floatImage;
    if (mat.depth() == CV_32F)
        floatImage = mat;
    else
        mat.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // Treat channels as extra floats per row
    const int channels = floatImage.channels();
    ImageData result = allocateGPUImage(floatImage.cols * channels, floatImage.rows);
    result.width = floatImage.cols;

    cudaMemcpy2D(result.gpuData, result.pitch, floatImage.data, floatImage.step[0],
        floatImage.cols * channels * sizeof(float), floatImage.rows, cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GPU copy failed: ") + cudaGetErrorString(err));
    }

    return result;
}

void ImageHandler::saveImageFromGPU(const ImageData& imageData, const std::string& filename, bool signedRange) {
    // Allocate host memory as contiguous array
    float* hostData = new float[imageData.width * imageData.height];

    try {
        // Copy from GPU (pitch-aware)
        copyGPUToHost(imageData, hostData);

        // Create cv::Mat from contiguous host data
        // Data is already flattened row-by-row, so create as contiguous
        cv::Mat floatImage(imageData.height, imageData.width, CV_32F);
        memcpy(floatImage.data, hostData, imageData.width * imageData.height * sizeof(float));

        // Convert to uint8 image
       // Convert to uint8 image
        cv::Mat uint8Image;
        if (signedRange)
            floatImage.convertTo(uint8Image, CV_8U, 127.5, 127.5); // [-1,1] -> [0,255]
        else
            floatImage.convertTo(uint8Image, CV_8U, 255.0f);
        // Save
        if (!cv::imwrite(filename, uint8Image)) {
            throw std::runtime_error("Failed to save image: " + filename);
        }
    }
    catch (...) {
        delete[] hostData;
        throw;
    }

    delete[] hostData;
}

ImageData ImageHandler::allocateGPUImage(int width, int height) {
    ImageData result;
    result.width = width;
    result.height = height;

    // Allocate with pitch for optimal memory alignment
    cudaMallocPitch(&result.gpuData, &result.pitch, width * sizeof(float), height);

    if (!result.gpuData) {
        throw std::runtime_error("Failed to allocate GPU memory for image");
    }

    return result;
}

void ImageHandler::freeGPUImage(ImageData& image) {
    if (image.gpuData) {
        cudaFree(image.gpuData);
        image.gpuData = nullptr;
    }
}

void ImageHandler::copyHostToGPU(const float* hostData, ImageData& image, size_t hostPitch) {
    cudaMemcpy2D(image.gpuData, image.pitch, hostData, hostPitch,
        image.width * sizeof(float), image.height, cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GPU copy failed: ") + cudaGetErrorString(err));
    }
}

void ImageHandler::copyGPUToHost(const ImageData& image, float* hostData) {
    cudaMemcpy2D(hostData, image.width * sizeof(float), image.gpuData, image.pitch,
        image.width * sizeof(float), image.height, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GPU copy failed: ") + cudaGetErrorString(err));
    }
}

cv::Mat ImageHandler::getImageFromGPU(const ImageData& imageData) {
    // Allocate host memory as contiguous array
    float* hostData = new float[imageData.width * imageData.height];

    try {
        // Copy from GPU (pitch-aware)
        copyGPUToHost(imageData, hostData);

        // Create cv::Mat from contiguous host data
        cv::Mat result(imageData.height, imageData.width, CV_32F);
        memcpy(result.data, hostData, imageData.width * imageData.height * sizeof(float));

        delete[] hostData;
        return result;
    }
    catch (...) {
        delete[] hostData;
        throw;
    }
}