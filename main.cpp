#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include "include/normal_permutation_generator.h"
#include "include/cloud_loader.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <array>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

const bool SAVE_DEBUG_DATA = false;
const bool REMOVE_SFM_POINTS = true;            // keep that on !!!
const int WINDOW_RADIUS = 3;                    // 3-5 works well << if clouds come out noisy then this is too small, if clouds come out "oversmoothed" then this is too large
const int DEPTH_STEPS = 16;                     // originally 16. Can be increased but not sure if it changes anything yet
const int NORMAL_PERMUTATIONS = 16;
const std::vector<std::pair<float, int>> NORMAL_PERMUTATION_RINGS = { {2.f, 7}, {4.f, 8} };
const float DEPTH_STEP_SIZE = 0.01f;            // there is a sweet spot for this. If clouds look "stepped" its not the sweet spot
const float BEST_CAMERA_ANGLE_DEGREES = 17.5f;  // dont know yet. in theory this should be 25-30deg but then in another pdf it said 5-30 
const int FRONTIER_MIN_SIZE = 100;              // early stop except 1 frontier
const int MAX_ITERATIONS = 1000;

// 16 : {{4.f, 7}, {14.f, 8}}
// 16 : {{3.f, 4}, {8.f, 5}, {18.f, 6}}
// 32 : {{3.f, 15}, {12.f, 16}}
// 32 : {{2.5f, 8}, {7.f, 10}, {18.f, 13}}
// 32 : {{2.f, 6}, {5.f, 7}, {10.f, 8}, {20.f, 10}}
// 32 : {{2.f, 4}, {4.f, 5}, {8.f, 6}, {14.f, 7}, {24.f, 9}}

// move helpers to another file
HomographyParams calcHomographyParamsW2C(const Camera& cam1, const Camera& cam2);
void saveNormalsAsXyz(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap, const Camera& cam);
void saveNormalMapPng(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap);
std::vector<float> generateDepthSteps(float interval, int steps);
std::unordered_map<int, std::unordered_map<int, float>> calculateCameraAxisAngles(const std::vector<Camera>& cameras);
std::vector<int> findBestCamerasByAngleGaussian(const int& refCamIdx, const int& NCams, const std::unordered_map<int, std::unordered_map<int, float>>& cameraAngle, const float& bestCameraAngleDegrees);

// TODO:
// - add color to dense point clouds
// - add more anlgles to normal permutations DONE
// - add normal vector dot product during propagation to filter out "impossible" normals
// - try different zncc aggregation methods
// - switch to disparity?
// - find the pdf with "chessboard pattern propagation" or "red-black" !!! (gipuma?) << https://openaccess.thecvf.com/content_iccv_2015/papers/Galliani_Massively_Parallel_Multiview_ICCV_2015_paper.pdf doesn't really solve anything
// - precompute normals in the sparse point cloud DONE-ish

int main() {    

    std::string folderPath = "data\\japcat9\\";
    std::vector<Camera> cameras;
    {
        CameraLoader cameraLoader;
        cameras = cameraLoader.loadFromCSV("data\\japcat9\\japcat.csv");
    }

    if (cameras.size() < 3) {
        std::cerr << "Error: At least 3 cameras are required for this algorithm." << std::endl;
        return -1;
	}

    std::unordered_map<int, std::unordered_map<int, float>> cameraAngle = calculateCameraAxisAngles(cameras);
    ImageHandler imageHandler;

    NormalPermutationGenerator normalPermGen(NORMAL_PERMUTATIONS);
    std::vector<NormalPermutation> normalPermutations = normalPermGen.generateMulti(NORMAL_PERMUTATION_RINGS);
	//std::vector<NormalPermutation> normalPermutations = { {0.0f, 0.0f} }; // only the parent normal, no permutations}

    if (SAVE_DEBUG_DATA) {
        normalPermGen.saveNormalsAsXyz("normals_debug.xyz", normalPermutations);
    }

    for (int refCamIdx = 0; refCamIdx < static_cast<int>(cameras.size()); ++refCamIdx) {

        auto bestCameras = findBestCamerasByAngleGaussian(refCamIdx, 2, cameraAngle, BEST_CAMERA_ANGLE_DEGREES);

        int targetIdx0 = -1;
        int targetIdx1 = -1;

        if (bestCameras.size() >= 2) {
            targetIdx0 = bestCameras[0];
            targetIdx1 = bestCameras[1];
        }
        else {
            std::cerr << "Not enough best cameras found for reference camera " + std::to_string(refCamIdx) << std::endl; // selection is fixed, this should never happen again
            return -1;
        }

        const std::string refImgName = cameras[refCamIdx].name;
        const std::string refImgPth = folderPath + "img\\" + refImgName;
        ImageData refImageGPU = imageHandler.loadImageToGPU(refImgPth);

        const std::string targetImgPth0 = folderPath + "img\\" + cameras[targetIdx0].name;
        const std::string targetImgPth1 = folderPath + "img\\" + cameras[targetIdx1].name;
        ImageData targetImage0GPU = imageHandler.loadImageToGPU(targetImgPth0);
        ImageData targetImage1GPU = imageHandler.loadImageToGPU(targetImgPth1);

        float avgDepth = 0.0f;
        CloudLoader cloudLoader;

        auto refDepthNormals = cloudLoader.loadXYZToDepthMapWithNormals("data\\japcat9\\japcat.xyz", cameras[refCamIdx], refImageGPU, true);

        float calculatedDepthStepSize = DEPTH_STEP_SIZE; // was (avgDepth * DEPTH_STEP_SIZE) but was unreliable << depth is not linear

        const int IMG_WIDTH = refImageGPU.width;
        const int IMG_HEIGHT = refImageGPU.height;
        const double progressImageScale = 900.f / float(std::max(IMG_WIDTH, IMG_HEIGHT));

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

        std::vector<float> depthSteps = generateDepthSteps(calculatedDepthStepSize, DEPTH_STEPS);

        HomographyParams hp0 = calcHomographyParamsW2C(cameras[refCamIdx], cameras[targetIdx0]);
        HomographyParams hp1 = calcHomographyParamsW2C(cameras[refCamIdx], cameras[targetIdx1]);

        processor.setCameraMatrices(hp0.H0.data(), hp0.K1inv.data(), hp0.e2.data());
        processor.setCameraMatricesSecond(hp1.H0.data(), hp1.K1inv.data(), hp1.e2.data());
        processor.setNormalPermutations(normalPermutations.data(), static_cast<int>(normalPermutations.size()));

        //cv::Mat depthMapHost = refDepthNormals.depthMap.clone();   // why am I using clones ?
		//cv::Mat normalMapHost = refDepthNormals.normalMap.clone(); // why am I using clones ? + the depth map is not used for anything other then debug
        cv::Mat visitedHost = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);

        std::vector<FrontierPoint> frontierPoints;
        frontierPoints.reserve(1024);           
        //cv::Vec3f initialNormal = cv::Vec3f(0.f, 0.f, 1.0f); // 0,0,1 turns normals away from cam
        cv::Vec3f initialNormal = cv::Vec3f(0.1f, 0.f, -0.9f); // tilted normal leaves the normal facing the cam 
        for (int y = 0; y < IMG_HEIGHT; ++y) {
            for (int x = 0; x < IMG_WIDTH; ++x) {
                float d = refDepthNormals.depthMap.at<float>(y, x);
                if (d > 0.f) {
					cv::Vec3f normal = refDepthNormals.normalMap.at<cv::Vec3f>(y, x);
                    FrontierPoint fp;
                    fp.x = x;
                    fp.y = y;
                    fp.depth = d;
                    fp.normal = make_float3(initialNormal[0], initialNormal[1], initialNormal[2]);
                    //fp.normal = make_float3(normal[0], normal[1], normal[2]);
                    frontierPoints.push_back(fp);
                    visitedHost.at<uchar>(y, x) = 1;
                    //normalMap.at<cv::Vec3f>(y, x) = initialNormal;
                }
            }
        }

        size_t iteration = 0;        

        while (!frontierPoints.empty() && iteration < MAX_ITERATIONS && frontierPoints.size() > FRONTIER_MIN_SIZE) {
            ++iteration;
            std::cout << "Iteration " << iteration << ", frontier size: " << frontierPoints.size() << "                \r" << std::flush;

            if (REMOVE_SFM_POINTS && iteration == 1) {
                refDepthNormals.depthMap.setTo(0.f);
            }

            std::vector<ResultPoint> results = processor.computeZNCCDualTargets(
                refImageGPU,
                targetImage0GPU,
                targetImage1GPU,
				refDepthNormals.depthMap, // hmm       
                frontierPoints,
                depthSteps,
                static_cast<int>(normalPermutations.size()));

            if (results.empty()) {
                std::cout << "No valid results returned by computeZNCC. Stopping.\n";
                break;
            }

            // update depth map, normal map and visited mask << there is some logic potential here like revisiting points with low score
            for (const auto& rp : results) {
                if (rp.y >= 0 && rp.y < IMG_HEIGHT && rp.x >= 0 && rp.x < IMG_WIDTH) {
                    refDepthNormals.depthMap.at<float>(rp.y, rp.x) = rp.depth;
                    refDepthNormals.normalMap.at<cv::Vec3f>(rp.y, rp.x) = cv::Vec3f(rp.normal.x, rp.normal.y, rp.normal.z);
                    visitedHost.at<uchar>(rp.y, rp.x) = 1;
                }
            }

            // build next frontier                       // could chcek if a resulting normal looks "possible" here
            std::vector<FrontierPoint> nextFrontier;
            nextFrontier.reserve(results.size() * 8);
            for (const auto& rp : results) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = rp.x + dx;
                        int ny = rp.y + dy;
                        if (nx < 0 || nx >= IMG_WIDTH || ny < 0 || ny >= IMG_HEIGHT) continue;
                        if (visitedHost.at<uchar>(ny, nx) == 0) {
                            float normalCamDot; // normal to camera dot product should reflect less then ~60deg (0.5)     float dot = normal[0] * 0.f + normal[1] * 0.f + normal[2] * -1.f; // or 1.f
                            FrontierPoint child;
                            child.x = nx;
                            child.y = ny;
                            child.depth = rp.depth;
                            child.normal = rp.normal;
                            nextFrontier.push_back(child);
                            visitedHost.at<uchar>(ny, nx) = 1;
                        }
                    }
                }
            }

            frontierPoints.swap(nextFrontier);

            if (true) { // i guess this isn't good for performance. the whole program is in debug state anyway
                cv::Mat progressImage;
                cv::resize(refDepthNormals.depthMap, progressImage, cv::Size(), progressImageScale, progressImageScale, cv::INTER_NEAREST);
                progressImage.convertTo(progressImage, CV_8U, 255.f, 0.0f);
                cv::imshow("Depth Map Progress", progressImage);
                cv::waitKey(1);
            }
        }

        cv::destroyAllWindows();        

        std::string pointCloudFilename = "clouds\\cloud_" + std::to_string(refCamIdx) + "_w" + std::to_string(1 + WINDOW_RADIUS * 2) + "_c" + std::to_string(8) + "_dss" + std::to_string(calculatedDepthStepSize) + ".xyz";
        saveNormalsAsXyz(pointCloudFilename, refDepthNormals.normalMap, refDepthNormals.depthMap, cameras[refCamIdx]);

        if (SAVE_DEBUG_DATA) {
            saveNormalMapPng("result_normal_map.png", refDepthNormals.normalMap, refDepthNormals.depthMap);
            cv::Mat depthMapToSave, depthMapNormalized;
            cv::normalize(refDepthNormals.depthMap, depthMapNormalized, 0, 1, cv::NORM_MINMAX);
            depthMapNormalized.convertTo(depthMapToSave, CV_8U, 255.0f);
            cv::imwrite("final_depth_map.png", depthMapToSave);
        }

        imageHandler.freeGPUImage(refImageGPU);
        imageHandler.freeGPUImage(targetImage0GPU);
        imageHandler.freeGPUImage(targetImage1GPU);

        std::cout << "\n\nReference Camera[" << refCamIdx << "] Iterations: " << iteration << "\n";
    }
    
    return 0;
}

HomographyParams calcHomographyParamsW2C( // finally...
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
    cv::Mat Rinv = R.t();                      // rotation inverse (world <- camera)

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
            double nx = static_cast<double>(n[0]);
            double ny = static_cast<double>(n[1]);
            double nz = static_cast<double>(n[2]);

            // normalize normal in camera frame if needed
            double len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-6) {
                nx /= len; ny /= len; nz /= len;
            }
            else {
                // fallback normal (camera frame)
                nx = 0.0; ny = 0.0; nz = 1.0;
            }

            // Rotate normal to world coordinates: n_w = R^T * n_c (Rinv == R.t())
            cv::Mat nC = (cv::Mat_<double>(3, 1) << nx, ny, nz);
            cv::Mat nW = Rinv * nC;

            double nwx = nW.at<double>(0, 0);
            double nwy = nW.at<double>(1, 0);
            double nwz = nW.at<double>(2, 0);

            // normalize world normal to be safe
            double nwlen = std::sqrt(nwx * nwx + nwy * nwy + nwz * nwz);
            if (nwlen > 1e-6) {
                nwx /= nwlen; nwy /= nwlen; nwz /= nwlen;
            }
            else {
                nwx = 0.0; nwy = 0.0; nwz = 1.0;
            }

            int nbytes = snprintf(lineBuf, sizeof(lineBuf), "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                Xw.at<double>(0, 0),
                Xw.at<double>(1, 0),
                Xw.at<double>(2, 0),
                static_cast<float>(nwx), static_cast<float>(nwy), static_cast<float>(nwz));
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

// Ray through pixel (x, y) in camera coordinates (unnormalized).
cv::Vec3d pixelRay(const cv::Mat& Kinv, double x, double y)
{
    return cv::Vec3d(
        Kinv.at<double>(0, 0) * x + Kinv.at<double>(0, 1) * y + Kinv.at<double>(0, 2),
        Kinv.at<double>(1, 0) * x + Kinv.at<double>(1, 1) * y + Kinv.at<double>(1, 2),
        Kinv.at<double>(2, 0) * x + Kinv.at<double>(2, 1) * y + Kinv.at<double>(2, 2));
}

// Plane offset d of the plane n·X = d (camera frame) through the 3D point of pixel (x, y) at 'depth'.
// Same convention as the kernel: pivot = ray * depth / ray.z, planeOffset = n · pivot.
float computePlaneOffset(const cv::Mat& Kinv, int x, int y, float depth, const cv::Vec3f& n)
{
    const cv::Vec3d r = pixelRay(Kinv, x, y);
    const double s = depth / r[2];
    return static_cast<float>(n[0] * r[0] * s + n[1] * r[1] * s + n[2] * r[2] * s);
}

void saveNormalMapPng(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap)
{
    if (normalMap.empty() || depthMap.empty()) {
        std::cerr << "Normal map or depth map is empty, cannot save PNG.\n";
        return;
    }
    cv::Mat vis;    
    normalMap.convertTo(vis, CV_32F);
    vis = (vis * 0.5f) + 0.5f;
    vis *= 255.0f;
    cv::Mat vis8;
    vis.convertTo(vis8, CV_8UC3);
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

std::unordered_map<int, std::unordered_map<int, float>> calculateCameraAxisAngles(const std::vector<Camera>& cameras) {    
    std::unordered_map<int, std::unordered_map<int, float>> cameraAngle;
    const cv::Mat z = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    const int n = static_cast<int>(cameras.size());
    for (int i = 0; i < n; ++i) {
        const cv::Mat R1 = cameras[i].extrinsics.rotation;
        cv::Mat dir1 = R1.t() * z;
        double n1 = cv::norm(dir1);
        for (int j = i + 1; j < n; ++j) {
            const cv::Mat R2 = cameras[j].extrinsics.rotation;
            cv::Mat dir2 = R2.t() * z;
            double n2 = cv::norm(dir2);
            double cosang = 0.0;
            if (n1 > 1e-12 && n2 > 1e-12) {
                cosang = (dir1.dot(dir2)) / (n1 * n2);
            }            
            cosang = std::max(-1.0, std::min(1.0, cosang));
            float angleDeg = static_cast<float>(std::acos(cosang) * 180.0 / CV_PI);
            cameraAngle[i][j] = angleDeg;
            cameraAngle[j][i] = angleDeg;
        }
    }
    return cameraAngle;
}

std::vector<int> findBestCamerasByAngleGaussian( // don't really need gaussian distribution for angles only but hopefully there will be more params in the future
    const int& refCamIdx,
    const int& NCams,
    const std::unordered_map<int, std::unordered_map<int, float>>& cameraAngle,
    const float& bestCameraAngleDegrees) {
    
    int allTCams = cameraAngle.at(refCamIdx).size();

    if (allTCams < NCams) throw("not enough tcams to satisfy request");

	std::vector<std::pair<float, int>> candidates; // score, camIdx
    candidates.reserve(allTCams);

    // score with gaussian distribution:
	// score = exp(-(camAngle - bestCameAngle)^2 / (2 * fct^2))     fct ~ 6 for smooth decay around 5 and 30 deg. this will only work for 17.5deg

    const float fct = 6.f;

    for (auto& tCam : cameraAngle.at(refCamIdx)) {
        int tCamIdx = tCam.first;
        float tCamAngle = tCam.second;
        float score = exp(-std::pow((tCamAngle - bestCameraAngleDegrees),2.f) / (2.f * fct * fct));        
        candidates.emplace_back(std::make_pair(score, tCamIdx));
	}    
    
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { // have to use lambda to prevent indices influencing sorting
            return a.first > b.first; 
        });

    std::vector<int> ret;
    ret.reserve(NCams);

    for (int i = 0; i < NCams; i++) ret.emplace_back(candidates[i].second);

    return ret;
}