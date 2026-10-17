#pragma once
#include <cuda_runtime.h> // just for float3... meh
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <vector>

// i should probably split this into HOST/GPU types so there is no openCV and CUDA dependency in the same header

struct NormalPermutation
{
    float tangent;
    float bitangent;
};

struct FrontierPoint
{
    int x;
    int y;
    float depth;   // from neighbor parent
    float3 normal; // from neighbor parent
};

struct ResultPoint
{
    int x;
    int y;
    float depth;
    float3 normal;
	float znccScore = -1.f;
};

struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t
};

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

// logic not yet implemented
//struct FolderData {
//    std::string inputFolder;
//    std::string csvFile;    
//    std::string sparseCloudFile;
//	std::vector<std::string> imageFiles;
//};