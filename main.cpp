#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include "include/normal_permutation_generator.h"
#include "include/cloud_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>

const int WINDOW_RADIUS = 2;
const int DEPTH_STEPS = 16;
const float coneAngleDegrees = 5.0f;

struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t
};

HomographyParams calcHomographyParams2(
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

std::vector<float> generateDepthSteps(float interval, int steps) { // its gonna be so much fun to debug it with disparity :/
	std::vector<float> depthSteps(steps); // hope this is enough to make it contiguous    
	int startIdx = -steps / 2;
    for (int i = 0; i < steps; ++i) {
        depthSteps[i] = (startIdx + i) * interval; // ooor I could make it into disparity here
	}
    return depthSteps;
}

int main() {    
    
    NormalPermutationGenerator normalPermGen(16);
    std::vector<NormalPermutation> normalPermutations = normalPermGen.generateSingle(coneAngleDegrees, true);
    normalPermGen.saveNormalsAsXyz("normals_debug.xyz", coneAngleDegrees);

    std::vector<Camera> cameras;

    {
        CameraLoader cameraLoader;
        cameras = cameraLoader.loadFromCSV("data\\budda2\\budda.csv");
    }

    ImageHandler imageHandler;

    ImageData refImage = imageHandler.loadImageToGPU("data\\budda2\\img\\41454.png"); // idx 0
    ImageData targetImage = imageHandler.loadImageToGPU("data\\budda2\\img\\41455.png"); // idx 1

    CloudLoader cloudLoader; // make static !
    cv::Mat depthMapInit = cloudLoader.loadXYZToDepthMap("data\\budda2\\budda.xyz", cameras[0], refImage);

    const int IMG_WIDTH = refImage.width;
    const int IMG_HEIGHT = refImage.height;

    ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

    std::vector<float> depthSteps = generateDepthSteps(0.01f, DEPTH_STEPS); // 0.01 for calcHomographyParams1 , 

    std::vector<FrontierPoint> frontierPoints;

    for (int y = 0; y < IMG_HEIGHT; ++y) {
        for (int x = 0; x < IMG_WIDTH; ++x) {
            float depth = depthMapInit.at<float>(y, x);
            if (depth > 0.f) { // valid depth
                frontierPoints.push_back({ x, y, depth, {0.f, 0.f, 1.f} }); // add normal init from sparse point cloud in the future
            }
        }
	}

    HomographyParams hp = calcHomographyParams2(cameras[0], cameras[1]);
    processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());
    processor.setNormalPermutations(normalPermutations.data(), normalPermutations.size());

    std::vector<ResultPoint> result = processor.computeZNCC(refImage, targetImage, frontierPoints, depthSteps, normalPermutations.size());

	depthMapInit.setTo(-1.f); // reset depth map to -1 for invalid points

    for (const auto& rp : result) {
		depthMapInit.at<float>(rp.y, rp.x) = rp.depth; // update depth map with new depth
	}

    imageHandler.freeGPUImage(refImage);
    imageHandler.freeGPUImage(targetImage);

    std::cout << "Done!\n";
    return 0;
   
}
