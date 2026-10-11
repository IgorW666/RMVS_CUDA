#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "camera_loader.h"
#include "image_handler.h"

/**
 * CloudLoader
 *
 * Loads an XYZ point cloud (whitespace separated x y z per line) and
 * creates a sparse depth map for a given camera/image size.
 *
 * Uses world->camera extrinsics: X_cam = R * X_world + t
 * Output depth map: CV_32FC1, depth value = Z (camera-space z), empty pixels = 0.0f
 */
    class CloudLoader {
    public:
        CloudLoader() = default;
        ~CloudLoader() = default;

        CloudLoader(const CloudLoader&) = delete;
        CloudLoader& operator=(const CloudLoader&) = delete;

        cv::Mat loadXYZToDepthMap(const std::string& filename, const Camera& cam, const ImageData& imageData, float& avgDepth);
        cv::Mat loadXYZToDepthMap(const std::string& filename, const Camera& cam, int width, int height, float& avgDepth);

    private:
        void projectPointAndUpdateDepth(const cv::Mat& Kd, const cv::Mat& R, const cv::Mat& t,
            const cv::Vec3d& point_world, std::vector<float>& depthBuffer,
            int width, int height);
};