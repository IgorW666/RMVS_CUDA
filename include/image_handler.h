#pragma once

#include <string>
#include <opencv2/opencv.hpp>

/**
 * Image data container for GPU/CPU transfers.
 */
struct ImageData {
    float* gpuData;      // GPU memory pointer
    int width;
    int height;
    size_t pitch;        // Pitch for GPU memory (bytes)

    ImageData() : gpuData(nullptr), width(0), height(0), pitch(0) {}
};

class ImageHandler {
public:
    ImageHandler() = default;
    ~ImageHandler();

    // Deleted copy operations (GPU memory management)
    ImageHandler(const ImageHandler&) = delete;
    ImageHandler& operator=(const ImageHandler&) = delete;

    /**
     * Load image from disk (OpenCV) and copy to GPU memory.
     * Converts to grayscale if needed.
     *
     * @param filename  Path to image file
     * @return          ImageData struct with GPU pointer
     */
    ImageData loadImageToGPU(const std::string& filename);

    /**
     * Load image from cv::Mat (OpenCV) and copy to GPU memory.
     * Converts to grayscale if needed.
     *
     * @param mat  OpenCV Mat containing image data
     * @return     ImageData struct with GPU pointer
     */
    ImageData loadImageToGPU(const cv::Mat& mat);

    /**
     * Copy GPU memory back to host and save as image file.
     *
     * @param imageData GPU image data
     * @param filename  Output file path
     */
    void saveImageFromGPU(const ImageData& imageData, const std::string& filename, bool signedRange = false);

    /**
     * Allocate GPU memory for image.
     *
     * @param width     Image width
     * @param height    Image height
     * @return          ImageData struct with allocated GPU memory
     */
    ImageData allocateGPUImage(int width, int height);

    /**
     * Free GPU memory associated with image.
     *
     * @param image     Image data to free
     */
    void freeGPUImage(ImageData& image);

    /**
     * Copy image data from host to GPU with pitch handling.
     *
     * @param hostData   Host pointer to image data
     * @param image      Target GPU image
     * @param hostPitch  Pitch of host data in bytes (row stride)
     */
    void copyHostToGPU(const float* hostData, ImageData& image, size_t hostPitch);

    /**
     * Copy image data from GPU to host.
     *
     * @param image     Source GPU image
     * @param hostData  Host pointer to receive data (width * height floats)
     */
    void copyGPUToHost(const ImageData& image, float* hostData);

    /**
 * Copy GPU memory back to host as cv::Mat (float32).
 *
 * @param imageData GPU image data
 * @return          cv::Mat with image data (CV_32F)
 */
    cv::Mat getImageFromGPU(const ImageData& imageData);

private:
};