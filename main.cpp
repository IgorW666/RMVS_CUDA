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
    //try {
	    const float coneAngleDegrees = 15.0f;
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
		cv::Mat depthMap0 = cloudLoader.loadXYZToDepthMap("data\\budda2\\budda.xyz", cameras[0], refImage);
        //cv::Mat depthMap1 = cloudLoader.loadXYZToDepthMap("data\\budda2\\budda.xyz", cameras[1], refImage);
        //cv::Mat depthMap2 = cloudLoader.loadXYZToDepthMap("data\\budda2\\budda.xyz", cameras[2], refImage);
        //cv::Mat depthMap3 = cloudLoader.loadXYZToDepthMap("data\\budda2\\budda.xyz", cameras[3], refImage);

        const int IMG_WIDTH = refImage.width;
        const int IMG_HEIGHT = refImage.height;        

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);	

		//FrontierPoint frontierPoint1 = {IMG_WIDTH / 2, IMG_HEIGHT / 2, 1.f, {0.0f, 0.0f, 1.0f}}; // 20.f for calcHomographyParams1
        //FrontierPoint frontierPoint2 = { IMG_WIDTH / 2 + 10, IMG_HEIGHT / 2, 20.f, {0.0f, 0.0f, 1.0f} };
		int fp3x = 1079; // 2079, 1145
		int fp3y = 2084;
		float fp3depth = depthMap0.at<float>(fp3y, fp3x); // 20.f for calcHomographyParams1
        FrontierPoint frontierPoint3 = { fp3x, fp3y, fp3depth, {0.f,0.f,1.f} };

		std::vector<float> depthSteps = generateDepthSteps(0.01f, DEPTH_STEPS); // 0.01 for calcHomographyParams1 , 

		std::vector<FrontierPoint> frontierPoints = { frontierPoint3 };

        HomographyParams hp = calcHomographyParams2(cameras[0], cameras[1]);
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
/*
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
*/