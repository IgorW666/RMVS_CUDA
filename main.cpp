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

// Save depth map as an XYZ point cloud in world coordinates.
// Writes one XYZ point per line for valid depth (> 0).
void saveDepthMapAsXyz(const std::string& filename, const cv::Mat& depthMap, const Camera& cam)
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    const cv::Mat K = cam.intrinsics.camera_matrix; // CV_64F
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    // Extrinsics: X_cam = R * X_world + t  =>  X_world = R^T * (X_cam - t)
    cv::Mat R = cam.extrinsics.rotation;       // 3x3 CV_64F
    cv::Mat t = cam.extrinsics.translation;    // 3x1 CV_64F
    cv::Mat Rinv = R.t();

    const int h = depthMap.rows;
    const int w = depthMap.cols;

    out << std::fixed << std::setprecision(6);
    for (int v = 0; v < h; ++v) {
        for (int u = 0; u < w; ++u) {
            float z = depthMap.at<float>(v, u);
            if (!(z > 0.0f)) continue; // skip invalid / negative depths

            // Back-project to camera coordinates
            const double Xc = (static_cast<double>(u) - cx) * static_cast<double>(z) / fx;
            const double Yc = (static_cast<double>(v) - cy) * static_cast<double>(z) / fy;
            const double Zc = static_cast<double>(z);

            cv::Mat Xc_mat = (cv::Mat_<double>(3, 1) << Xc, Yc, Zc);
            cv::Mat Xw = Rinv * (Xc_mat - t);

            out << Xw.at<double>(0, 0) << " "
                << Xw.at<double>(1, 0) << " "
                << Xw.at<double>(2, 0) << "\n";
        }
    }

    out.close();
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

	const double progressImageScale = 0.25; // will have to calculate this from screen resolution and image resolution

    ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

    std::vector<float> depthSteps = generateDepthSteps(0.01f, DEPTH_STEPS); // 0.01 for calcHomographyParams1 , 

    HomographyParams hp = calcHomographyParams2(cameras[0], cameras[1]);
    processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());
    processor.setNormalPermutations(normalPermutations.data(), normalPermutations.size());

    // Prepare working depth map and visited mask
    cv::Mat depthMap = depthMapInit.clone(); // we'll update this in-place as frontier grows
    cv::Mat visited = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U); // 0 = not visited/assigned, 1 = assigned or queued

    // Initialize visited mask from existing sparse depth map
    std::vector<FrontierPoint> frontierPoints;
    for (int y = 0; y < IMG_HEIGHT; ++y) {
        for (int x = 0; x < IMG_WIDTH; ++x) {
            float depth = depthMap.at<float>(y, x);
            if (depth > 0.f) { // valid depth
                FrontierPoint fp;
                fp.x = x;
                fp.y = y;
                fp.depth = depth;
                fp.normal = { 0.f, 0.f, 1.f };
                frontierPoints.push_back(fp);
                visited.at<uchar>(y, x) = 1;
            }
        }
    }

    // Frontier growing loop: expand around survived ResultPoints until no new frontier points can be built
    size_t iteration = 0;
    const size_t maxIterations = static_cast<size_t>(IMG_WIDTH) * static_cast<size_t>(IMG_HEIGHT); // safety cap

    while (!frontierPoints.empty() && iteration < maxIterations) {
        ++iteration;
        std::cout << "Iteration " << iteration << ", frontier size: " << frontierPoints.size() << std::endl;

        // Run ZNCC kernel for current frontier
        std::vector<ResultPoint> results = processor.computeZNCC(refImage, targetImage, frontierPoints, depthSteps, static_cast<int>(normalPermutations.size()));

        if (results.empty()) {
            std::cout << "No valid results returned by computeZNCC. Stopping.\n";
            break;
        }

        // Update depth map and mark visited for survived points
        for (const auto& rp : results) {
            // Update depth - this will overwrite existing value or fill a new one
            if (rp.y >= 0 && rp.y < IMG_HEIGHT && rp.x >= 0 && rp.x < IMG_WIDTH) {
                depthMap.at<float>(rp.y, rp.x) = rp.depth;
                visited.at<uchar>(rp.y, rp.x) = 1;
            }
        }

        // Build next frontier around each survived ResultPoint (8-connected neighbors)
        std::vector<FrontierPoint> nextFrontier;
        nextFrontier.reserve(results.size() * 8);

        for (const auto& rp : results) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = rp.x + dx;
                    int ny = rp.y + dy;
                    if (nx < 0 || nx >= IMG_WIDTH || ny < 0 || ny >= IMG_HEIGHT) continue;

                    // Only add neighbor if it hasn't been visited/queued yet
                    if (visited.at<uchar>(ny, nx) == 0) {
                        FrontierPoint child;
                        child.x = nx;
                        child.y = ny;
                        child.depth = rp.depth;      // inherit depth from parent
                        child.normal = rp.normal;    // inherit normal from parent
                        nextFrontier.push_back(child);
                        visited.at<uchar>(ny, nx) = 1; // mark as queued/visited to prevent duplicates
                    }
                }
            }
        }

        std::cout << "Valid results: " << results.size() << ", next frontier size: " << nextFrontier.size() << std::endl;

        // Move to next frontier
        frontierPoints.swap(nextFrontier);

		cv::Mat progressImage;
		cv::resize(depthMap, progressImage, cv::Size(), progressImageScale, progressImageScale, cv::INTER_NEAREST);
		progressImage.convertTo(progressImage, CV_8U, 255.0f); // scale depth to 0-255 for visualization
		cv::imshow("Depth Map Progress", progressImage);
		cv::waitKey(1); // small delay to allow window to update
    }

    // At this point depthMap contains the grown depths; if needed, save or further process.
    // For backward compatibility with earlier code which expected depthMapInit updated:
    depthMapInit = depthMap;
    saveDepthMapAsXyz("result_grown.xyz", depthMapInit, cameras[0]);

	// Save final depth map to disk
	cv::Mat depthMapToSave, depthMapNormalized;
	cv::normalize(depthMap, depthMapNormalized, 0, 1, cv::NORM_MINMAX); // normalize to 0-1 for visualization
	depthMapNormalized.convertTo(depthMapToSave, CV_8U, 255.0f); // scale depth to 0-255 for saving
	cv::imwrite("final_depth_map.png", depthMapToSave);

    imageHandler.freeGPUImage(refImage);
    imageHandler.freeGPUImage(targetImage);

    std::cout << "Done! Iterations: " << iteration << "\n";
    return 0;

}