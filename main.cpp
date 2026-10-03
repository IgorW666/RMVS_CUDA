#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>

struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t
};

// Plane-induced homography: H = K2 * (R + t * n^T / d) * K1^-1
// p' ~ H0 * p + (K2 * t) * (n^T * K1^-1 * p) / d
HomographyParams calcHomographyParams(const Camera& cam1, const Camera& cam2) {
    cv::Mat K1 = cam1.intrinsics.camera_matrix;
    cv::Mat K2 = cam2.intrinsics.camera_matrix;
    cv::Mat R = cam2.extrinsics.rotation * cam1.extrinsics.rotation.t();
    //cv::Mat t = cam2.extrinsics.translation - R * cam1.extrinsics.translation;
    cv::Mat t = cam2.extrinsics.rotation * (cam1.extrinsics.translation - cam2.extrinsics.translation);

    cv::Mat K1inv = K1.inv();
    cv::Mat H0 = K2 * R * K1inv;
    cv::Mat e = K2 * t; // 3x1

    HomographyParams params;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            params.H0[i * 3 + j] = static_cast<float>(H0.at<double>(i, j));
            params.K1inv[i * 3 + j] = static_cast<float>(K1inv.at<double>(i, j));
        }
        params.e2[i] = static_cast<float>(e.at<double>(i, 0));
    }

    return params;
}

int main() {
    try {
        // Initialize
        ImageHandler imageHandler;
        CameraLoader cameraLoader;

        auto cameras = cameraLoader.loadFromCSV("data\\crazy.csv");

        for (int i : { 0, 6 }) {
            const Camera& c = cameras[i];
            std::cout << c.name << " tx,ty,tz = " << c.extrinsics.translation.t()
                << " f=" << c.intrinsics.camera_matrix.at<double>(0, 0)
                << " px,py=" << c.intrinsics.camera_matrix.at<double>(0, 2) << ","
                << c.intrinsics.camera_matrix.at<double>(1, 2) << "\n";
        }
        cv::Mat dbgR = cameras[6].extrinsics.rotation * cameras[0].extrinsics.rotation.t();
        cv::Mat dbgT = cameras[6].extrinsics.rotation *
            (cameras[0].extrinsics.translation - cameras[6].extrinsics.translation);
        std::cout << "baseline |t| = " << cv::norm(dbgT) << "\n";

        // Load images FIRST to get actual dimensions
        std::cout << "Loading images...\n";
        ImageData refImage = imageHandler.loadImageToGPU("data\\img\\P1000965.png"); // idx 0
        ImageData targetImage = imageHandler.loadImageToGPU("data\\img\\P1000971.png"); // idx 6

        // Use ACTUAL image dimensions
        const int IMG_WIDTH = refImage.width;
        const int IMG_HEIGHT = refImage.height;
        const int WINDOW_RADIUS = 2;

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

        




            // Load depth and normal maps with ACTUAL dimensions
            //cv::Mat depthDummy(IMG_HEIGHT, IMG_WIDTH, CV_32F, cv::Scalar(5.0f));
            float f = static_cast<float>(cameras[0].intrinsics.camera_matrix.at<double>(0, 0));
            const float DUMMY_DEPTH = static_cast<float>(f * cv::norm(dbgT) / 50.0); // f from camera 0
            cv::Mat depthDummy(IMG_HEIGHT, IMG_WIDTH, CV_32F, cv::Scalar(DUMMY_DEPTH));
            ImageData depthMap = imageHandler.loadImageToGPU(depthDummy);

            cv::Mat normalDummy(IMG_HEIGHT, IMG_WIDTH, CV_32FC3, cv::Scalar(0.0f, 0.0f, 1.0f));
            ImageData normalMap = imageHandler.loadImageToGPU(normalDummy);

            HomographyParams hp = calcHomographyParams(cameras[0], cameras[6]);
            processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());

            // Compute ZNCC
            std::cout << "Computing ZNCC...\n";
            ImageData result = processor.computeZNCC(refImage, targetImage, depthMap, normalMap);

            // Save results (ZNCC is in [-1, 1])
            std::cout << "Saving results...\n";
            imageHandler.saveImageFromGPU(result, "zncc_output.png", true);
        

        // Cleanup
        imageHandler.freeGPUImage(refImage);
        imageHandler.freeGPUImage(targetImage);
        imageHandler.freeGPUImage(depthMap);
        imageHandler.freeGPUImage(normalMap);

        std::cout << "Done!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}