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
#include <cstdio> // for snprintf

const bool SAVE_DEBUG_DATA = false;
const bool REMOVE_SFM_POINTS = true;
const int WINDOW_RADIUS = 3;
const int DEPTH_STEPS = 16;
const float CONE_ANGLE_DEGREES = 2.0f;
const float DEPTH_STEP_SIZE = 0.005f;

// move to types.h
struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t
};

// move helpers to another file
HomographyParams calcHomographyParams2(const Camera& cam1, const Camera& cam2);
void saveNormalsAsXyz(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap, const Camera& cam);
void saveNormalMapPng(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap);
std::vector<float> generateDepthSteps(float interval, int steps);

int main() {

    NormalPermutationGenerator normalPermGen(16);
    std::vector<NormalPermutation> normalPermutations = normalPermGen.generateSingle(CONE_ANGLE_DEGREES, true);

    if (SAVE_DEBUG_DATA) {
        normalPermGen.saveNormalsAsXyz("normals_debug.xyz", CONE_ANGLE_DEGREES);
    }

    std::vector<Camera> cameras;

    {
        CameraLoader cameraLoader;
        cameras = cameraLoader.loadFromCSV("data\\budda2\\budda.csv");
    }

    ImageHandler imageHandler;

    ImageData refImage = imageHandler.loadImageToGPU("data\\budda2\\img\\41455.png"); // idx 0
    ImageData targetImage = imageHandler.loadImageToGPU("data\\budda2\\img\\41456.png"); // idx 1

    CloudLoader cloudLoader; // make static !
    cv::Mat depthMapInit = cloudLoader.loadXYZToDepthMap("data\\budda2\\budda.xyz", cameras[1], refImage);

    const int IMG_WIDTH = refImage.width;
    const int IMG_HEIGHT = refImage.height;

    const double progressImageScale = 0.25; // will have to calculate this from screen resolution and image resolution

    ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

    std::vector<float> depthSteps = generateDepthSteps(DEPTH_STEP_SIZE, DEPTH_STEPS); // 0.01 for calcHomographyParams1 , 

    HomographyParams hp = calcHomographyParams2(cameras[1], cameras[2]);
    processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());
    processor.setNormalPermutations(normalPermutations.data(), normalPermutations.size());

    // Prepare working depth map, normal map and visited mask
    cv::Mat depthMap = depthMapInit.clone(); // we'll update this in-place as frontier grows
    cv::Mat normalMap = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32FC3); // per-pixel normals (x,y,z)
    cv::Mat visited = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U); // 0 = not visited/assigned, 1 = assigned or queued

    // Initialize visited mask and normal map from existing sparse depth map
    std::vector<FrontierPoint> frontierPoints;
    for (int y = 0; y < IMG_HEIGHT; ++y) {
        for (int x = 0; x < IMG_WIDTH; ++x) {
            float depth = depthMap.at<float>(y, x);
            if (depth > 0.f) { // valid depth
                FrontierPoint fp;
                fp.x = x;
                fp.y = y;
                fp.depth = depth;
                fp.normal = make_float3(0.f, 0.f, 1.f);
                frontierPoints.push_back(fp);
                visited.at<uchar>(y, x) = 1;
                normalMap.at<cv::Vec3f>(y, x) = cv::Vec3f(0.f, 0.f, 1.f);
            }
        }
    }

    // Frontier growing loop: expand around survived ResultPoints until no new frontier points can be built
    size_t iteration = 0;
    const size_t maxIterations = static_cast<size_t>(IMG_WIDTH) * static_cast<size_t>(IMG_HEIGHT); // safety cap

    while (!frontierPoints.empty() && iteration < maxIterations) {
        ++iteration;
        std::cout << "Iteration " << iteration << ", frontier size: " << frontierPoints.size() << std::endl;

        // Remove original SFM points once before running ZNCC so they won't be re-added to the results
        if (REMOVE_SFM_POINTS && iteration == 1) {
            depthMap.setTo(0.f);
        }

        // Run ZNCC kernel for current frontier
        std::vector<ResultPoint> results = processor.computeZNCC(refImage, targetImage, frontierPoints, depthSteps, static_cast<int>(normalPermutations.size()));

        if (results.empty()) {
            std::cout << "No valid results returned by computeZNCC. Stopping.\n";
            break;
        }

        // Update depth map, normal map and mark visited for survived points
        for (const auto& rp : results) {
            // Update depth & normal - this will overwrite existing value or fill a new one
            if (rp.y >= 0 && rp.y < IMG_HEIGHT && rp.x >= 0 && rp.x < IMG_WIDTH) {
                depthMap.at<float>(rp.y, rp.x) = rp.depth;
                // rp.normal is float3 (from types.h)
                normalMap.at<cv::Vec3f>(rp.y, rp.x) = cv::Vec3f(rp.normal.x, rp.normal.y, rp.normal.z);
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

        frontierPoints.swap(nextFrontier);

        cv::Mat progressImage;
        cv::resize(depthMap, progressImage, cv::Size(), progressImageScale, progressImageScale, cv::INTER_NEAREST);
        progressImage.convertTo(progressImage, CV_8U, 255.0f); // scale depth to 0-255 for visualization
        cv::imshow("Depth Map Progress", progressImage);
        cv::waitKey(1);
    }

    cv::destroyAllWindows();
    depthMapInit = depthMap;    

	std::string pointCloudFilename = "cloud_w" + std::to_string(1 + WINDOW_RADIUS * 2) + "_c" + std::to_string(CONE_ANGLE_DEGREES) + "_dss" + std::to_string(DEPTH_STEP_SIZE) + ".xyz";

    saveNormalsAsXyz(pointCloudFilename, normalMap, depthMapInit, cameras[1]);

    if (SAVE_DEBUG_DATA) {
        saveNormalMapPng("result_normal_map.png", normalMap, depthMapInit);
        cv::Mat depthMapToSave, depthMapNormalized;
        cv::normalize(depthMap, depthMapNormalized, 0, 1, cv::NORM_MINMAX); // normalize to 0-1 for visualization
        depthMapNormalized.convertTo(depthMapToSave, CV_8U, 255.0f); // scale depth to 0-255 for saving
        cv::imwrite("final_depth_map.png", depthMapToSave);
	}
    
    imageHandler.freeGPUImage(refImage);
    imageHandler.freeGPUImage(targetImage);

    std::cout << "Done! Iterations: " << iteration << "\n";
    return 0;

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

// Save normal map into XYZ file (x y z nx ny nz), using depth map to backproject points.
void saveNormalsAsXyz(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap, const Camera& cam)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    const cv::Mat K = cam.intrinsics.camera_matrix; // CV_64F
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    cv::Mat R = cam.extrinsics.rotation;       // 3x3 CV_64F
    cv::Mat t = cam.extrinsics.translation;    // 3x1 CV_64F
    cv::Mat Rinv = R.t();

    const int h = depthMap.rows;
    const int w = depthMap.cols;

    const size_t FLUSH_THRESHOLD = 1 << 20; // 1 MB
    std::string buffer;
    buffer.reserve(FLUSH_THRESHOLD + 256);

    char lineBuf[256];
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

            cv::Vec3f n = normalMap.at<cv::Vec3f>(v, u);
            float nx = n[0], ny = n[1], nz = n[2];
            // normalize normal if needed
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-6f) {
                nx /= len; ny /= len; nz /= len;
            }
            else {
                // fallback normal
                nx = 0.f; ny = 0.f; nz = 1.f;
            }

            int nbytes = snprintf(lineBuf, sizeof(lineBuf), "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                Xw.at<double>(0, 0),
                Xw.at<double>(1, 0),
                Xw.at<double>(2, 0),
                nx, ny, nz);
            if (nbytes > 0) {
                buffer.append(lineBuf, static_cast<size_t>(nbytes));
            }

            if (buffer.size() >= FLUSH_THRESHOLD) {
                out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                buffer.clear();
            }
        }
    }

    if (!buffer.empty()) {
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    }
    out.close();
}

// Save normal map as a 3-channel PNG for visualization. Valid pixels are taken from depthMap mask.
void saveNormalMapPng(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap)
{
    if (normalMap.empty() || depthMap.empty()) {
        std::cerr << "Normal map or depth map is empty, cannot save PNG.\n";
        return;
    }
    cv::Mat vis;
    // Map normals from [-1,1] to [0,255]
    normalMap.convertTo(vis, CV_32F);
    vis = (vis * 0.5f) + 0.5f; // now in [0,1] (assumes normals roughly in [-1,1])
    vis *= 255.0f;
    cv::Mat vis8;
    vis.convertTo(vis8, CV_8UC3);

    // Zero-out pixels where depth <= 0 to indicate invalid areas
    for (int y = 0; y < depthMap.rows; ++y) {
        const float* dptr = depthMap.ptr<float>(y);
        uchar* vptr = vis8.ptr<uchar>(y);
        for (int x = 0; x < depthMap.cols; ++x) {
            if (!(dptr[x] > 0.0f)) {
                vptr[3 * x + 0] = 0;
                vptr[3 * x + 1] = 0;
                vptr[3 * x + 2] = 0;
            }
        }
    }

    if (!cv::imwrite(filename, vis8)) {
        std::cerr << "Failed to write normal PNG: " << filename << "\n";
    }
}

std::vector<float> generateDepthSteps(float interval, int steps) { // its gonna be so much fun to debug it with disparity :/
    std::vector<float> depthSteps(steps); // hope this is enough to make it contiguous    
    int startIdx = -steps / 2;
    for (int i = 0; i < steps; ++i) {
        depthSteps[i] = (startIdx + i) * interval; // ooor I could make it into disparity here
    }
    return depthSteps;
}