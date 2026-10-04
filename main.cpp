#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <iomanip>
#include <sstream>


const int WINDOW_RADIUS = 3;


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

void updatePlaneSweepResult(cv::Mat& planeSweepResult, cv::Mat& currentResult, float currentDepth, cv::Mat& depthResult)
{
	const int rows = planeSweepResult.rows;
	const int cols = planeSweepResult.cols;
	const float threshold = 0.75f; // ZNCC threshold for valid matches
    for (int r = 0; r < rows; r++) {
#pragma omp parallel for
        for (int c = 0; c < cols; c++) {
            float currentValue = currentResult.at<float>(r, c);
            float& bestValue = planeSweepResult.at<float>(r, c);
			float currentDepthValue = depthResult.at<float>(r, c);
            // Update if the current value is better (higher ZNCC score)
            if (currentValue > bestValue) {
                bestValue = currentValue;
                if (currentValue > threshold) { // Only update depth if ZNCC score is above threshold
                    depthResult.at<float>(r, c) = currentDepth;
                }
            }
		}
    }
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
        ImageData targetImage = imageHandler.loadImageToGPU("data\\img\\P1000966.png"); // idx 6

        // Use ACTUAL image dimensions
        const int IMG_WIDTH = refImage.width;
        const int IMG_HEIGHT = refImage.height;
        

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

        float f = static_cast<float>(cameras[0].intrinsics.camera_matrix.at<double>(0, 0));
        const float BASE_DEPTH = static_cast<float>(f * cv::norm(dbgT) / 50.0); // f from camera 0

        // Depth loop parameters
        const float DEPTH_MIN = BASE_DEPTH * 0.01f;
        const float DEPTH_MAX = BASE_DEPTH * 0.50f;
        const int DEPTH_STEPS = 10000;

        HomographyParams hp = calcHomographyParams(cameras[0], cameras[1]);
        processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());

        cv::Mat normalDummy(IMG_HEIGHT, IMG_WIDTH, CV_32FC3, cv::Scalar(0.0f, 0.0f, -1.0f));
        ImageData normalMap = imageHandler.loadImageToGPU(normalDummy);

        cv::Mat planeSweepResult(IMG_HEIGHT, IMG_WIDTH, CV_32FC1, cv::Scalar(0.f));
        cv::Mat depthResult(IMG_HEIGHT, IMG_WIDTH, CV_32FC1, cv::Scalar(0.f));

        // Loop over different depth values
        for (int depthStep = 0; depthStep < DEPTH_STEPS; ++depthStep) {
            float depth = DEPTH_MIN + (DEPTH_MAX - DEPTH_MIN) * depthStep / (DEPTH_STEPS - 1);

            //depth = 1 / depth; 

            std::cout << "Processing depth " << depthStep + 1 << "/" << DEPTH_STEPS
                << " (depth = " << depth << ")...\n";

            // Create depth map with current depth value
            cv::Mat depthDummy(IMG_HEIGHT, IMG_WIDTH, CV_32F, cv::Scalar(depth));
            ImageData depthMap = imageHandler.loadImageToGPU(depthDummy);

            // Compute ZNCC
            ImageData result = processor.computeZNCC(refImage, targetImage, depthMap, normalMap);

            // Generate output filename with depth index
            std::ostringstream oss;
            oss << "znccRes/zncc_output_depth_" << std::setfill('0') << std::setw(2) << depthStep << ".png";
            std::string outputFilename = oss.str();

            // Save results (ZNCC is in [-1, 1])
            std::cout << "Saving " << outputFilename << "\n\n\n";
            //imageHandler.saveImageFromGPU(result, outputFilename.c_str(), true);

            cv::Mat resultHost = imageHandler.getImageFromGPU(result); // I need a cv::Mat version of result here

			updatePlaneSweepResult(planeSweepResult, resultHost, depth, depthResult);

            // Cleanup for this iteration
            imageHandler.freeGPUImage(depthMap);
        }

        imageHandler.freeGPUImage(normalMap);

        // Cleanup
        imageHandler.freeGPUImage(refImage);
        imageHandler.freeGPUImage(targetImage);

		// save final plane sweep result and depth map as CV_8U images (normalized)
		cv::Mat planeSweepResultNormalized, depthResultNormalized;
		cv::normalize(planeSweepResult, planeSweepResultNormalized, 0, 255 , cv::NORM_MINMAX);
		cv::normalize(depthResult, depthResultNormalized, 0, 255 , cv::NORM_MINMAX);
		cv::imwrite("plane_sweep_result.png", planeSweepResultNormalized);
		cv::imwrite("depth_result.png", depthResultNormalized);


        std::cout << "Done!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}