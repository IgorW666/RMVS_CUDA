
#include "../include/cloud_loader.h"

#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>

cv::Mat CloudLoader::loadXYZToDepthMap(const std::string& filename, const Camera& cam, const ImageData& imageData, float& avgDepth) {
    return loadXYZToDepthMap(filename, cam, imageData.width, imageData.height, avgDepth);
}

cv::Mat CloudLoader::loadXYZToDepthMap(const std::string& filename, const Camera& cam, int width, int height, float& avgDepth) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width and height must be positive");
    }

    std::vector<float> depthBuffer(static_cast<size_t>(width) * static_cast<size_t>(height),
        std::numeric_limits<float>::infinity());

	avgDepth = 0.0f;

    // Prepare intrinsics and extrinsics once (converted to CV_64F)
    cv::Mat Kd;
    if (cam.intrinsics.camera_matrix.type() == CV_64F) Kd = cam.intrinsics.camera_matrix;
    else cam.intrinsics.camera_matrix.convertTo(Kd, CV_64F);

    cv::Mat R;
    if (cam.extrinsics.rotation.type() == CV_64F) R = cam.extrinsics.rotation;
    else cam.extrinsics.rotation.convertTo(R, CV_64F);

    cv::Mat t;
    if (cam.extrinsics.translation.type() == CV_64F) t = cam.extrinsics.translation;
    else cam.extrinsics.translation.convertTo(t, CV_64F);

    // Ensure t is 3x1
    if (t.rows == 1 && t.cols == 3) t = t.t();

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open point cloud file: " + filename);
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        // skip comments
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) continue;

        cv::Vec3d pw(x, y, z); // world point
        projectPointAndUpdateDepth(Kd, R, t, pw, depthBuffer, width, height);
    }

    // Convert depthBuffer to cv::Mat (CV_32FC1); set empty (inf) -> 0.0f
    cv::Mat depthMap(height, width, CV_32FC1);
    size_t idx = 0;
    float depthSum = 0.0f;
    size_t depthCount = 0;
    for (int r = 0; r < height; ++r) {
        float* rowPtr = depthMap.ptr<float>(r);
        for (int c = 0; c < width; ++c, ++idx) {
            float d = depthBuffer[idx];
            rowPtr[c] = std::isfinite(d) ? d : 0.0f;
            if (std::isfinite(d)) {
                depthSum += d;
                ++depthCount;
            }
        }
    }

	avgDepth = (depthCount > 0) ? (depthSum / static_cast<float>(depthCount)) : 0.0f;

    return depthMap;
}

void CloudLoader::projectPointAndUpdateDepth(const cv::Mat& Kd, const cv::Mat& R, const cv::Mat& t,
    const cv::Vec3d& point_world, std::vector<float>& depthBuffer,
    int width, int height) {
    // X_cam = R * X_world + t
    cv::Mat pw = (cv::Mat_<double>(3, 1) << point_world[0], point_world[1], point_world[2]);
    cv::Mat Xc = R * pw + t; // 3x1

    double X = Xc.at<double>(0, 0);
    double Y = Xc.at<double>(1, 0);
    double Z = Xc.at<double>(2, 0);

    if (!(Z > 0.0)) return; // behind camera or on focal plane

    // Project to pixel coords
    cv::Mat proj = Kd * (Xc / Z); // 3x1
    double u = proj.at<double>(0, 0);
    double v = proj.at<double>(1, 0);

    int ui = static_cast<int>(std::round(u));
    int vi = static_cast<int>(std::round(v));

    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;

    size_t idx = static_cast<size_t>(vi) * static_cast<size_t>(width) + static_cast<size_t>(ui);

    float zf = static_cast<float>(Z);
    // Keep the closest point => smallest positive Z
    if (zf < depthBuffer[idx]) depthBuffer[idx] = zf;
}