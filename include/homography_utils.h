#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <cmath>


HomographyParams calcHomographyParamsW2C( // finally...
    const Camera& cam1,
    const Camera& cam2) {

    // Intrinsics
    cv::Mat K1 = cam1.intrinsics.camera_matrix;
    cv::Mat K2 = cam2.intrinsics.camera_matrix;

    // Assume extrinsics are world->camera: X_cam = R * X_world + t
    cv::Mat R1 = cam1.extrinsics.rotation;   // world->camera rotation for cam1
    cv::Mat R2 = cam2.extrinsics.rotation;   // world->camera rotation for cam2
    cv::Mat t1 = cam1.extrinsics.translation; // world->camera translation for cam1 (3x1)
    cv::Mat t2 = cam2.extrinsics.translation; // world->camera translation for cam2 (3x1)

    // Relative pose: transform points from camera1 frame to camera2 frame
    // X_c2 = R2 * R1^T * X_c1 + (t2 - R2 * R1^T * t1)
    cv::Mat R_rel = R2 * R1.t();
    cv::Mat t_rel = t2 - R_rel * t1;

    cv::Mat K1inv = K1.inv();
    cv::Mat H0 = K2 * R_rel * K1inv;
    cv::Mat e2 = K2 * t_rel;

    HomographyParams params;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            params.H0[i * 3 + j] = static_cast<float>(H0.at<double>(i, j));
            params.K1inv[i * 3 + j] = static_cast<float>(K1inv.at<double>(i, j));
        }
        params.e2[i] = static_cast<float>(e2.at<double>(i, 0));
    }
    return params;
}

// Ray through pixel (x, y) in camera coordinates (unnormalized).
cv::Vec3d pixelRay(const cv::Mat& Kinv, double x, double y)
{
    return cv::Vec3d(
        Kinv.at<double>(0, 0) * x + Kinv.at<double>(0, 1) * y + Kinv.at<double>(0, 2),
        Kinv.at<double>(1, 0) * x + Kinv.at<double>(1, 1) * y + Kinv.at<double>(1, 2),
        Kinv.at<double>(2, 0) * x + Kinv.at<double>(2, 1) * y + Kinv.at<double>(2, 2));
}

// Plane offset d of the plane n·X = d (camera frame) through the 3D point of pixel (x, y) at 'depth'.
// Same convention as the kernel: pivot = ray * depth / ray.z, planeOffset = n · pivot.
float computePlaneOffset(const cv::Mat& Kinv, int x, int y, float depth, const cv::Vec3f& n)
{
    const cv::Vec3d r = pixelRay(Kinv, x, y);
    const double s = depth / r[2];
    return static_cast<float>(n[0] * r[0] * s + n[1] * r[1] * s + n[2] * r[2] * s);
}

std::vector<float> generateDepthSteps(float interval, int steps) { // its gonna be so much fun to debug it with disparity :/
    std::vector<float> depthSteps(steps); // hope this is enough to make it contiguous    
    int startIdx = -steps / 2;
    for (int i = 0; i < steps; ++i) {
        depthSteps[i] = (startIdx + i) * interval; // ooor I could make it into disparity here
    }
    return depthSteps;
}

std::unordered_map<int, std::unordered_map<int, float>> calculateCameraAxisAngles(const std::vector<Camera>& cameras) {
    std::unordered_map<int, std::unordered_map<int, float>> cameraAngle;
    const cv::Mat z = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    const int n = static_cast<int>(cameras.size());
    for (int i = 0; i < n; ++i) {
        const cv::Mat R1 = cameras[i].extrinsics.rotation;
        cv::Mat dir1 = R1.t() * z;
        double n1 = cv::norm(dir1);
        for (int j = i + 1; j < n; ++j) {
            const cv::Mat R2 = cameras[j].extrinsics.rotation;
            cv::Mat dir2 = R2.t() * z;
            double n2 = cv::norm(dir2);
            double cosang = 0.0;
            if (n1 > 1e-12 && n2 > 1e-12) {
                cosang = (dir1.dot(dir2)) / (n1 * n2);
            }
            cosang = std::max(-1.0, std::min(1.0, cosang));
            float angleDeg = static_cast<float>(std::acos(cosang) * 180.0 / CV_PI);
            cameraAngle[i][j] = angleDeg;
            cameraAngle[j][i] = angleDeg;
        }
    }
    return cameraAngle;
}

std::vector<int> findBestCamerasByAngleGaussian( // don't really need gaussian distribution for angles only but hopefully there will be more params in the future
    const int& refCamIdx,
    const int& NCams,
    const std::unordered_map<int, std::unordered_map<int, float>>& cameraAngle,
    const float& bestCameraAngleDegrees) {

    int allTCams = cameraAngle.at(refCamIdx).size();

    if (allTCams < NCams) std::cerr << "not enough tcams to satisfy request\n";

    std::vector<std::pair<float, int>> candidates; // score, camIdx
    candidates.reserve(allTCams);

    // score with gaussian distribution:
    // score = exp(-(camAngle - bestCameAngle)^2 / (2 * fct^2))     fct ~ 6 for smooth decay around 5 and 30 deg. this will only work for 17.5deg

    const float fct = 6.f;

    for (auto& tCam : cameraAngle.at(refCamIdx)) {
        int tCamIdx = tCam.first;
        float tCamAngle = tCam.second;
        float score = exp(-std::pow((tCamAngle - bestCameraAngleDegrees), 2.f) / (2.f * fct * fct));
        candidates.emplace_back(std::make_pair(score, tCamIdx));
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { // have to use lambda to prevent indices influencing sorting
            return a.first > b.first;
        });

    std::vector<int> ret;
    ret.reserve(NCams);

    for (int i = 0; i < NCams; i++) ret.emplace_back(candidates[i].second);

    return ret;
}