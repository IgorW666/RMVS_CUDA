#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include "camera_loader.h"
#include "image_handler.h"

/**
 * DepthNormalData
 *
 * Output structure containing depth map and optional normal map.
 * depthMap: CV_32FC1, depth value = Z (camera-space z), empty pixels = 0.0f
 * normalMap: CV_32FC3, normal vectors in camera space (x, y, z), empty pixels = (0, 0, 0)
 * avgDepth: average depth of valid pixels
 */
struct DepthNormalData {
    cv::Mat depthMap;
    cv::Mat normalMap;
    float avgDepth;
};

/**
 * CloudLoader
 *
 * Loads an XYZ point cloud (whitespace separated x y z per line) and
 * creates a sparse depth map with optional normal vectors for a given camera/image size.
 *
 * Uses world->camera extrinsics: X_cam = R * X_world + t
 * Optionally computes surface normals via KNN + PCA and filters by camera-facing orientation.
 */
class CloudLoader {
public:
    CloudLoader() = default;
    ~CloudLoader() = default;

    CloudLoader(const CloudLoader&) = delete;
    CloudLoader& operator=(const CloudLoader&) = delete;

    // Load depth map only (backward compatible)
    cv::Mat loadXYZToDepthMap(const std::string& filename, const Camera& cam, const ImageData& imageData, float& avgDepth);
    cv::Mat loadXYZToDepthMap(const std::string& filename, const Camera& cam, int width, int height, float& avgDepth);

    // Load depth map with optional normal computation
    DepthNormalData loadXYZToDepthMapWithNormals(const std::string& filename, const Camera& cam,
        const ImageData& imageData, bool precomputeNormals = false);
    DepthNormalData loadXYZToDepthMapWithNormals(const std::string& filename, const Camera& cam,
        int width, int height, bool precomputeNormals = false);

private:
    struct SurvivingPoint {
        cv::Vec3d posCamera;     // position in camera space
        int pixelX, pixelY;      // projected image coordinates
        float depth;             // Z in camera space
    };

    void projectPointAndUpdateDepth(const cv::Mat& Kd, const cv::Mat& R, const cv::Mat& t,
        const cv::Vec3d& point_world, std::vector<float>& depthBuffer,
        int width, int height);

    std::vector<SurvivingPoint> filterAndProjectPoints(const std::string& filename,
        const cv::Mat& Kd, const cv::Mat& R, const cv::Mat& t,
        int width, int height);

    cv::Vec3f computeNormalViaPCA(const std::vector<cv::Vec3d>& neighbors);

    bool isNormalFacingCamera(const cv::Vec3f& normal, float dotThreshold = std::cos(60.0f * 3.14159265f / 180.0f));

    static const int KNN_K = 6;
    static const float NORMAL_FACING_THRESHOLD;  // cos(60 degrees)
};