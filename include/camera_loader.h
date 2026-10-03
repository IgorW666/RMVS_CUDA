#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct CameraIntrinsics {
    cv::Mat camera_matrix;   // 3x3 intrinsic matrix K (CV_64F)
    cv::Mat distortion;      // Distortion coefficients (CV_64F, 1xN or Nx1)
};

struct CameraExtrinsics {
    cv::Mat translation;     // 3x1 translation vector (CV_64F)
    cv::Mat rotation;        // 3x3 rotation matrix (row-major, CV_64F)
};

struct Camera {
    std::string name;
    CameraExtrinsics extrinsics;
    CameraIntrinsics intrinsics;
};

class CameraLoader {
public:
    /// Load cameras from CSV file
    /// @param csv_path Path to the CSV file containing camera parameters
    /// @return Vector of Camera structures with loaded data
    static std::vector<Camera> loadFromCSV(const std::string& csv_path);
};