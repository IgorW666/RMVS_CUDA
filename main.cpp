#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include "include/normal_permutation_generator.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>

const int WINDOW_RADIUS = 2; // window size will be 2 * WINDOW_RADIUS + 1
const int DEPTH_STEPS = 16;

struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t
};

// Looks like this is the correct one
// plane induced homography: H = K2 * (R + t * n^T / d) * K1^-1
// p' ~ H0 * p + (K2 * t) * (n^T * K1^-1 * p) / d
HomographyParams calcHomographyParams1( // in case tx,ty,tz are camera centers, not translation vectors
    const Camera& cam1,
    const Camera& cam2){
    cv::Mat K1 = cam1.intrinsics.camera_matrix;
    cv::Mat K2 = cam2.intrinsics.camera_matrix;
    // RealityScan R assumed to be camera -> world.
    cv::Mat R = cam2.extrinsics.rotation * cam1.extrinsics.rotation.t();
    // RealityScan translation interpreted as camera center C
    // in world coordinates.
    cv::Mat C1 = cam1.extrinsics.translation;
    cv::Mat C2 = cam2.extrinsics.translation;    
    cv::Mat t = cam2.extrinsics.rotation * (C1 - C2);
    cv::Mat K1inv = K1.inv();
    cv::Mat H0 = K2 * R * K1inv;
    cv::Mat e2 = K2 * t;
    HomographyParams params;
    for (int i = 0; i < 3; ++i)    {
        for (int j = 0; j < 3; ++j)        {
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
    //try {
	    const float coneAngleDegrees = 15.0f; // 5 degrees cone angle
		NormalPermutationGenerator normalPermGen(16);
		std::vector<NormalPermutation> normalPermutations = normalPermGen.generateSingle(coneAngleDegrees, true); // 5 degrees cone angle        
        normalPermGen.saveNormalsAsXyz("normals_debug.xyz", coneAngleDegrees);

		std::vector<Camera> cameras;

        {
            CameraLoader cameraLoader;
			cameras = cameraLoader.loadFromCSV("data\\budda\\budda.csv");
        }

        ImageHandler imageHandler;
        
        ImageData refImage = imageHandler.loadImageToGPU("data\\budda\\img\\41454.jpg"); // idx 0
        ImageData targetImage = imageHandler.loadImageToGPU("data\\budda\\img\\41455.jpg"); // idx 1

        const int IMG_WIDTH = refImage.width;
        const int IMG_HEIGHT = refImage.height;        

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);	

		FrontierPoint frontierPoint1 = {IMG_WIDTH / 2, IMG_HEIGHT / 2, 20.f, {0.0f, 0.0f, 1.0f}};
        FrontierPoint frontierPoint2 = { IMG_WIDTH / 2 + 10, IMG_HEIGHT / 2, 20.f, {0.0f, 0.0f, 1.0f} };

		std::vector<float> depthSteps = generateDepthSteps(0.01f, DEPTH_STEPS);

		std::vector<FrontierPoint> frontierPoints = { frontierPoint1 };

        HomographyParams hp = calcHomographyParams1(cameras[0], cameras[1]);
        processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());
		processor.setNormalPermutations(normalPermutations.data(), normalPermutations.size());

        ImageData result = processor.computeZNCC(refImage, targetImage, frontierPoints, depthSteps, normalPermutations.size());
        
        cv::Mat resultHost = imageHandler.getImageFromGPU(result);

        //imageHandler.freeGPUImage(depthMap);              
        //imageHandler.freeGPUImage(normalMap);
        imageHandler.freeGPUImage(refImage);
        imageHandler.freeGPUImage(targetImage);

        std::cout << "Done!\n";
        return 0;
    //}
    //catch (const std::exception& e) {
    //    std::cerr << "Error: " << e.what() << "\n";
    //    return 1;
    //}
}

HomographyParams calcHomographyParams2( // in case tx,ty,tz are translation vectors, not camera centers
    const Camera& cam1,
    const Camera& cam2) {

    // debug print extrinsics
    if (1) {
        std::cout << "Camera 1 extrinsics:\n";
        std::cout << "Rotation:\n" << cam1.extrinsics.rotation << "\n";
        std::cout << "Translation:\n" << cam1.extrinsics.translation << "\n";
        std::cout << "Camera 2 extrinsics:\n";
        std::cout << "Rotation:\n" << cam2.extrinsics.rotation << "\n";
        std::cout << "Translation:\n" << cam2.extrinsics.translation << "\n";
    }

    cv::Mat K1 = cam1.intrinsics.camera_matrix;
    cv::Mat K2 = cam2.intrinsics.camera_matrix;
    cv::Mat R = cam2.extrinsics.rotation * cam1.extrinsics.rotation.t();  // relative rotation: camera 1 -> camera 2
    cv::Mat t1 = cam1.extrinsics.translation;
    cv::Mat t2 = cam2.extrinsics.translation;
    cv::Mat t = 2 - R * t1;  // relative translation
    cv::Mat K1inv = K1.inv();
    cv::Mat H0 = K2 * R * K1inv;
    cv::Mat e2 = K2 * t;
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