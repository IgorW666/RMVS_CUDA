#include "../include/cloud_loader.h"

#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>

const float CloudLoader::NORMAL_FACING_THRESHOLD = 0.5f;  // cos(60 degrees) ~ 0.5

cv::Mat CloudLoader::loadXYZToDepthMap(const std::string& filename, const Camera& cam, const ImageData& imageData) {
    return loadXYZToDepthMap(filename, cam, imageData.width, imageData.height);
}

cv::Mat CloudLoader::loadXYZToDepthMap(const std::string& filename, const Camera& cam, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width and height must be positive");
    }

    std::vector<float> depthBuffer(static_cast<size_t>(width) * static_cast<size_t>(height),
        std::numeric_limits<float>::infinity());

    avgDepth_ = 0.0f;
    minDepth_ = std::numeric_limits<float>::max();
    maxDepth_ = std::numeric_limits<float>::lowest();
    cv::Mat Kd;
    if (cam.intrinsics.camera_matrix.type() == CV_64F) Kd = cam.intrinsics.camera_matrix;
    else cam.intrinsics.camera_matrix.convertTo(Kd, CV_64F);
    cv::Mat R;
    if (cam.extrinsics.rotation.type() == CV_64F) R = cam.extrinsics.rotation;
    else cam.extrinsics.rotation.convertTo(R, CV_64F);
    cv::Mat t;
    if (cam.extrinsics.translation.type() == CV_64F) t = cam.extrinsics.translation;
    else cam.extrinsics.translation.convertTo(t, CV_64F);    
    if (t.rows == 1 && t.cols == 3) t = t.t();
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open point cloud file: " + filename);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) continue;

        cv::Vec3d pw(x, y, z);
        projectPointAndUpdateDepth(Kd, R, t, pw, depthBuffer, width, height);
    }
    cv::Mat depthMap(height, width, CV_32FC1);
    size_t idx = 0;
    float depthSum = 0.0f;
    size_t depthCount = 0;
    for (int r = 0; r < height; ++r) {
        float* rowPtr = depthMap.ptr<float>(r);
        for (int c = 0; c < width; ++c, ++idx) {
            float d = depthBuffer[idx];
            rowPtr[c] = std::isfinite(d) ? d : 0.0f;
            if (std::isfinite(d)) {
                depthSum += d;
                ++depthCount;
                minDepth_ = std::min(minDepth_, d);
                maxDepth_ = std::max(maxDepth_, d);
            }
        }
    }
    avgDepth_ = (depthCount > 0) ? (depthSum / static_cast<float>(depthCount)) : 0.0f;
    minDepth_ = (depthCount > 0) ? minDepth_ : 0.0f;
    maxDepth_ = (depthCount > 0) ? maxDepth_ : 0.0f;
    return depthMap;
}

DepthNormalData CloudLoader::loadXYZToDepthMapWithNormals(const std::string& filename, const Camera& cam, const ImageData& imageData, bool precomputeNormals) {
    return loadXYZToDepthMapWithNormals(filename, cam, imageData.width, imageData.height, precomputeNormals);
}

DepthNormalData CloudLoader::loadXYZToDepthMapWithNormals(const std::string& filename, const Camera& cam, int width, int height, bool precomputeNormals) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("width and height must be positive");
    }

    cv::Mat Kd;
    if (cam.intrinsics.camera_matrix.type() == CV_64F) Kd = cam.intrinsics.camera_matrix;
    else cam.intrinsics.camera_matrix.convertTo(Kd, CV_64F);

    cv::Mat R;
    if (cam.extrinsics.rotation.type() == CV_64F) R = cam.extrinsics.rotation;
    else cam.extrinsics.rotation.convertTo(R, CV_64F);

    cv::Mat t;
    if (cam.extrinsics.translation.type() == CV_64F) t = cam.extrinsics.translation;
    else cam.extrinsics.translation.convertTo(t, CV_64F);

    if (t.rows == 1 && t.cols == 3) t = t.t();

    std::vector<SurvivingPoint> survivingPoints = filterAndProjectPoints(filename, Kd, R, t, width, height);

    DepthNormalData result;
    result.depthMap = cv::Mat(height, width, CV_32FC1, 0.0f);
    result.normalMap = cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    //result.avgDepth = 0.0f;

    if (survivingPoints.empty()) {
        return result;
    }	

    std::vector<cv::Vec3f> normalMap(survivingPoints.size());

    if (precomputeNormals && survivingPoints.size() > KNN_K) {
        
        cv::Mat pointsForKNN(survivingPoints.size(), 3, CV_32F);
        for (size_t i = 0; i < survivingPoints.size(); ++i) {
            const auto& pt = survivingPoints[i];
            pointsForKNN.at<float>(i, 0) = static_cast<float>(pt.posCamera[0]);
            pointsForKNN.at<float>(i, 1) = static_cast<float>(pt.posCamera[1]);
            pointsForKNN.at<float>(i, 2) = static_cast<float>(pt.posCamera[2]);
        }

        cv::flann::Index flannIndex(pointsForKNN, cv::flann::KDTreeIndexParams());

        for (size_t i = 0; i < survivingPoints.size(); ++i) {
            cv::Mat query = pointsForKNN.row(i);
            cv::Mat indices, distances;
            flannIndex.knnSearch(query, indices, distances, KNN_K + 1, cv::flann::SearchParams());

            std::vector<cv::Vec3d> neighbors;
            for (int j = 1; j <= KNN_K && j < indices.cols; ++j) {
                int neighborIdx = indices.at<int>(0, j);
                neighbors.push_back(survivingPoints[neighborIdx].posCamera);
            }

            cv::Vec3f normal = computeNormalViaPCA(neighbors);

            if (isNormalFacingCamera(normal, NORMAL_FACING_THRESHOLD)) {
                normalMap[i] = normal;
            }
            else {                
                normalMap[i] = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }
        }
    }

    float depthSum = 0.0f;      
    float minDepth = std::numeric_limits<float>::max();
    float maxDepth = std::numeric_limits<float>::lowest();
	std::vector<float> depthValues; // for median
    size_t validCount = 0;

    for (size_t i = 0; i < survivingPoints.size(); ++i) {
        const auto& pt = survivingPoints[i];

        if (precomputeNormals) {
            const auto& normal = normalMap[i];
            if (normal[0] == 0.0f && normal[1] == 0.0f && normal[2] == 0.0f) {
                continue;
            }
        }

        float* depthPtr = result.depthMap.ptr<float>(pt.pixelY);
        depthPtr[pt.pixelX] = pt.depth;

        if (precomputeNormals) {
            cv::Vec3f* normalPtr = result.normalMap.ptr<cv::Vec3f>(pt.pixelY);
            normalPtr[pt.pixelX] = normalMap[i];
        }

        depthSum += pt.depth;
        minDepth = std::min(minDepth, pt.depth);
        maxDepth = std::max(maxDepth, pt.depth);
		depthValues.push_back(pt.depth);
        ++validCount;
    }

    minDepth_ = minDepth;
    maxDepth_ = maxDepth;
    avgDepth_ = (validCount > 0) ? (depthSum / static_cast<float>(validCount)) : 0.0f;
     
	int middleIndex = depthValues.size() / 2;

    medianDepth_ = (depthValues.empty()) ? 0.0f : (depthValues.size() % 2 == 0) ? (depthValues[middleIndex - 1] + depthValues[middleIndex]) / 2.0f : depthValues[middleIndex];

    return result;
}

std::vector<CloudLoader::SurvivingPoint> CloudLoader::filterAndProjectPoints(const std::string& filename,
    const cv::Mat& Kd, const cv::Mat& R, const cv::Mat& t,
    int width, int height) {
    std::vector<SurvivingPoint> result;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open point cloud file: " + filename);
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        double x, y, z;
        if (!(iss >> x >> y >> z)) continue;

        cv::Vec3d pw(x, y, z);

        cv::Mat pw_mat = (cv::Mat_<double>(3, 1) << pw[0], pw[1], pw[2]);
        cv::Mat Xc = R * pw_mat + t;

        double X = Xc.at<double>(0, 0);
        double Y = Xc.at<double>(1, 0);
        double Z = Xc.at<double>(2, 0);

        if (!(Z > 0.0)) continue;

        cv::Mat proj = Kd * (Xc / Z);
        double u = proj.at<double>(0, 0);
        double v = proj.at<double>(1, 0);

        int ui = static_cast<int>(std::round(u));
        int vi = static_cast<int>(std::round(v));

        if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;

        SurvivingPoint sp;
        sp.posCamera = cv::Vec3d(X, Y, Z);
        sp.pixelX = ui;
        sp.pixelY = vi;
        sp.depth = static_cast<float>(Z);
        result.push_back(sp);
    }

    return result;
}

cv::Vec3f CloudLoader::computeNormalViaPCA(const std::vector<cv::Vec3d>& neighbors) {
    if (neighbors.empty()) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }    
    cv::Vec3d mean(0.0, 0.0, 0.0);
    for (const auto& pt : neighbors) {
        mean += pt;
    }
    mean /= static_cast<double>(neighbors.size());

    cv::Mat cov = cv::Mat::zeros(3, 3, CV_64F);
    for (const auto& pt : neighbors) {
        cv::Vec3d centered = pt - mean;
        cv::Mat col = (cv::Mat_<double>(3, 1) << centered[0], centered[1], centered[2]);
        cov += col * col.t();
    }
    cov /= static_cast<double>(neighbors.size());

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    cv::Vec3f normal;
    normal[0] = static_cast<float>(eigenvectors.at<double>(2, 0));
    normal[1] = static_cast<float>(eigenvectors.at<double>(2, 1));
    normal[2] = static_cast<float>(eigenvectors.at<double>(2, 2));

    float len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (len > 1e-6f) {
        normal /= len;
    }

    return normal;
}

bool CloudLoader::isNormalFacingCamera(const cv::Vec3f& normal, float dotThreshold) {    
    cv::Vec3f cameraDir(0.0f, 0.0f, -1.0f); // -Z ?
    float dot = normal[0] * cameraDir[0] + normal[1] * cameraDir[1] + normal[2] * cameraDir[2];
    return dot > dotThreshold;
}

void CloudLoader::projectPointAndUpdateDepth(const cv::Mat& Kd, const cv::Mat& R, const cv::Mat& t,
    const cv::Vec3d& point_world, std::vector<float>& depthBuffer,
    int width, int height) {
    cv::Mat pw = (cv::Mat_<double>(3, 1) << point_world[0], point_world[1], point_world[2]);
    cv::Mat Xc = R * pw + t;

    double X = Xc.at<double>(0, 0);
    double Y = Xc.at<double>(1, 0);
    double Z = Xc.at<double>(2, 0);

    if (!(Z > 0.0)) return;

    cv::Mat proj = Kd * (Xc / Z);
    double u = proj.at<double>(0, 0);
    double v = proj.at<double>(1, 0);

    int ui = static_cast<int>(std::round(u));
    int vi = static_cast<int>(std::round(v));

    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;

    size_t idx = static_cast<size_t>(vi) * static_cast<size_t>(width) + static_cast<size_t>(ui);

    float zf = static_cast<float>(Z);
    if (zf < depthBuffer[idx]) depthBuffer[idx] = zf;
}