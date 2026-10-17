#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cmath>

// Save normal map into XYZ file (x y z nx ny nz), using depth map to backproject points.
void saveCloudsAsXyz(const std::string& filename, const cv::Mat& normalMap, const cv::Mat& depthMap, const Camera& cam, bool RGB)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    cv::Mat img;
    if (RGB) {
        // need absolute paths first
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