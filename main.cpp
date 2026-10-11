#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <iomanip>
#include <sstream>
#include <fstream>

const int WINDOW_RADIUS = 4;

struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t    
};

// Plane-induced homography: H = K2 * (R + t * n^T / d) * K1^-1
// p' ~ H0 * p + (K2 * t) * (n^T * K1^-1 * p) / d
/*HomographyParams calcHomographyParams( // in case tx,ty,tz are camera centers, not translation vectors
    const Camera& cam1,
    const Camera& cam2)
{
    cv::Mat K1 = cam1.intrinsics.camera_matrix;
    cv::Mat K2 = cam2.intrinsics.camera_matrix;

    // RealityScan R assumed to be camera -> world.
    cv::Mat R =
        cam2.extrinsics.rotation *
        cam1.extrinsics.rotation.t();

    // RealityScan translation interpreted as camera center C
    // in world coordinates.
    cv::Mat C1 = cam1.extrinsics.translation;
    cv::Mat C2 = cam2.extrinsics.translation;

    // Relative translation expressed in camera 2 coordinates.
    cv::Mat t =
        cam2.extrinsics.rotation *
        (C1 - C2);

    cv::Mat K1inv = K1.inv();

    cv::Mat H0 = K2 * R * K1inv;
    cv::Mat e2 = K2 * t;

    HomographyParams params;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            params.H0[i * 3 + j] =
                static_cast<float>(H0.at<double>(i, j));

            params.K1inv[i * 3 + j] =
                static_cast<float>(K1inv.at<double>(i, j));
        }

        params.e2[i] =
            static_cast<float>(e2.at<double>(i, 0));
    }

    return params;
}*/

HomographyParams calcHomographyParams( // in case tx,ty,tz are translation vectors, not camera centers
    const Camera& cam1,
    const Camera& cam2)
{
    cv::Mat K1 = cam1.intrinsics.camera_matrix;
    cv::Mat K2 = cam2.intrinsics.camera_matrix;

    // Relative rotation: camera 1 -> camera 2.
    cv::Mat R =
        cam2.extrinsics.rotation *
        cam1.extrinsics.rotation.t();

    // RealityScan translation interpreted as the
    // conventional OpenCV extrinsic t in:
    //
    // X_camera = R * X_world + t
    //
    cv::Mat t1 = cam1.extrinsics.translation;
    cv::Mat t2 = cam2.extrinsics.translation;

    // Relative translation.
    cv::Mat t =
        t2 - R * t1;

    cv::Mat K1inv = K1.inv();

    cv::Mat H0 = K2 * R * K1inv;
    cv::Mat e2 = K2 * t;

    HomographyParams params;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            params.H0[i * 3 + j] =
                static_cast<float>(H0.at<double>(i, j));

            params.K1inv[i * 3 + j] =
                static_cast<float>(K1inv.at<double>(i, j));
        }

        params.e2[i] =
            static_cast<float>(e2.at<double>(i, 0));
    }

    return params;
}

void updatePlaneSweepResult(cv::Mat& planeSweepResult, cv::Mat& currentResult, float currentDepth, cv::Mat& depthResult)
{
	const int rows = planeSweepResult.rows;
	const int cols = planeSweepResult.cols;
	const float threshold = 0.8f; // ZNCC threshold for valid matches
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

void updatePlaneSweepResultWithDepthRegularization(cv::Mat& planeSweepResult, cv::Mat& currentResult, float currentDepth, cv::Mat& depthResult)
{
    const int rows = planeSweepResult.rows;
    const int cols = planeSweepResult.cols;
    const float threshold = 0.65f;
    const float depthTolerance = 0.01f; // 10% tolerance for depth similarity
    const int minNeighbors = 1; // Minimum non-zero neighbors required

    for (int r = 0; r < rows; r++) {
#pragma omp parallel for
        for (int c = 0; c < cols; c++) {
            float currentValue = currentResult.at<float>(r, c);
            float& bestValue = planeSweepResult.at<float>(r, c);

            if (currentValue > bestValue && currentValue > threshold) {
                // Collect non-zero neighbor depths (8-connected)
                std::vector<float> neighborDepths;
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) continue; // Skip center pixel

                        int nr = r + dr;
                        int nc = c + dc;

                        // Check bounds
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                            float neighborDepth = depthResult.at<float>(nr, nc);
                            if (neighborDepth > 0.0f) {
                                neighborDepths.push_back(neighborDepth);
                            }
                        }
                    }
                }

                // If no neighbors yet, accept the update (first pass)
                if (neighborDepths.empty()) {
                    bestValue = currentValue;
                    depthResult.at<float>(r, c) = currentDepth;
                }
                // If neighbors exist, check similarity
                else if (neighborDepths.size() >= minNeighbors) {
                    // Calculate average neighbor depth
                    float avgNeighborDepth = 0.0f;
                    for (float d : neighborDepths) {
                        avgNeighborDepth += d;
                    }
                    avgNeighborDepth /= neighborDepths.size();

                    // Check if current depth is within tolerance of average neighbor depth
                    float depthDiff = std::abs(currentDepth - avgNeighborDepth);
                    float tolerance = avgNeighborDepth * depthTolerance;

                    if (depthDiff <= tolerance) {
                        bestValue = currentValue;
                        depthResult.at<float>(r, c) = currentDepth;
                    }
                }
            }
        }
    }
}

void saveDepthAsPointCloud(const cv::Mat& depthResult, const cv::Mat& K, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    float fx = static_cast<float>(K.at<double>(0, 0));
    float fy = static_cast<float>(K.at<double>(1, 1));
    float cx = static_cast<float>(K.at<double>(0, 2));
    float cy = static_cast<float>(K.at<double>(1, 2));

    for (int y = 0; y < depthResult.rows; ++y) {
        for (int x = 0; x < depthResult.cols; ++x) {
            float depth = depthResult.at<float>(y, x);

            // Skip zero depth values
            if (depth == 0.0f) {
                continue;
            }

            // Backproject pixel to 3D using camera intrinsics
            float Z = depth;
            float X = (x - cx) * Z / fx;
            float Y = (y - cy) * Z / fy;

            file << std::fixed << std::setprecision(6)
                << X << " " << Y << " " << Z << "\n";
        }
    }

    file.close();
}

int main() {
    try {
        // Initialize
        ImageHandler imageHandler;
        CameraLoader cameraLoader;

        auto cameras = cameraLoader.loadFromCSV("data\\budda\\budda.csv");

        /*for (int i : { 0, 6 }) {
            const Camera& c = cameras[i];
            std::cout << c.name << " tx,ty,tz = " << c.extrinsics.translation.t()
                << " f=" << c.intrinsics.camera_matrix.at<double>(0, 0)
                << " px,py=" << c.intrinsics.camera_matrix.at<double>(0, 2) << ","
                << c.intrinsics.camera_matrix.at<double>(1, 2) << "\n";
        }*/

        cv::Mat dbgR = cameras[1].extrinsics.rotation * cameras[0].extrinsics.rotation.t();
        cv::Mat dbgT = cameras[1].extrinsics.rotation *
            (cameras[0].extrinsics.translation - cameras[1].extrinsics.translation);
        std::cout << "baseline |t| = " << cv::norm(dbgT) << "\n";

        // Load images FIRST to get actual dimensions
        std::cout << "Loading images...\n";
        ImageData refImage = imageHandler.loadImageToGPU("data\\budda\\img\\41454.jpg"); // idx 0
        ImageData targetImage = imageHandler.loadImageToGPU("data\\budda\\img\\41457.jpg"); // idx 6

        // Use ACTUAL image dimensions
        const int IMG_WIDTH = refImage.width;
        const int IMG_HEIGHT = refImage.height;        

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

        float f = static_cast<float>(cameras[0].intrinsics.camera_matrix.at<double>(0, 0));
        const float BASE_DEPTH = static_cast<float>(f * cv::norm(dbgT) / 50.0); // f from camera 0

		// Depth loop parameters (budda dataset: 0.005 to 0.5 * BASE_DEPTH, 500 steps)
		const float DEPTH_MIN = BASE_DEPTH * 0.005f;  // depth min is actually disparity max (1/depth)
		const float DEPTH_MAX = BASE_DEPTH * 5.5f;    // depth max is actually disparity min (1/depth)
        const int DEPTH_STEPS = 1500;

        HomographyParams hp = calcHomographyParams(cameras[0], cameras[3]);
        processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());

        cv::Mat normalDummy(IMG_HEIGHT, IMG_WIDTH, CV_32FC3, cv::Scalar(0.0f, 0.0f, 1.0f));
        ImageData normalMap = imageHandler.loadImageToGPU(normalDummy);

        cv::Mat planeSweepResult(IMG_HEIGHT, IMG_WIDTH, CV_32FC1, cv::Scalar(0.f));
        cv::Mat depthResult(IMG_HEIGHT, IMG_WIDTH, CV_32FC1, cv::Scalar(0.f));

        // Loop over different depth values
        for (int depthStep = 0; depthStep < DEPTH_STEPS; ++depthStep) {
            // Calculate disparity (1/depth) linearly, then convert back to depth
            float disparity = (1.0f / DEPTH_MAX) + (1.0f / DEPTH_MIN - 1.0f / DEPTH_MAX) * depthStep / (DEPTH_STEPS - 1);
            float depth = 1.0f / disparity;

            //depth = 1 / depth; 

            std::cout << "Processing depth " << depthStep + 1 << "/" << DEPTH_STEPS
                << " (depth = " << depth << ", disparity = " << disparity << ")...\n";

            // Create depth map with current depth value
            cv::Mat depthDummy(IMG_HEIGHT, IMG_WIDTH, CV_32F, cv::Scalar(depth));
            ImageData depthMap = imageHandler.loadImageToGPU(depthDummy);

            // Compute ZNCC
            ImageData result = processor.computeZNCC(refImage, targetImage, depthMap, normalMap);

            if (false) {
                // Generate output filename with depth index
                std::ostringstream oss;
                oss << "znccRes/zncc_output_depth_" << std::setfill('0') << std::setw(2) << depthStep << ".png";
                std::string outputFilename = oss.str();

                // Save results (ZNCC is in [-1, 1])
                std::cout << "Saving " << outputFilename << "\n\n\n";
                imageHandler.saveImageFromGPU(result, outputFilename.c_str(), true);
            }

            cv::Mat resultHost = imageHandler.getImageFromGPU(result); // I need a cv::Mat version of result here

            updatePlaneSweepResult(planeSweepResult, resultHost, depth, depthResult);
			//updatePlaneSweepResultWithDepthRegularization(planeSweepResult, resultHost, depth, depthResult);

            // Cleanup for this iteration
            imageHandler.freeGPUImage(depthMap);
        }

        imageHandler.freeGPUImage(normalMap);

        // Cleanup
        imageHandler.freeGPUImage(refImage);
        imageHandler.freeGPUImage(targetImage);

        // Save depth result as point cloud (XYZ format)
        saveDepthAsPointCloud(depthResult, cameras[0].intrinsics.camera_matrix, "final_depth_result.xyz");

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