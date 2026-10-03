#include "../include/camera_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<Camera> CameraLoader::loadFromCSV(const std::string& csv_path) {
    std::vector<Camera> cameras;
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open camera CSV file: " + csv_path);
    }

    std::string line;

    // Skip header line
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty");
    }

    // Parse data lines
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Split CSV by commas
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        // Expected 22 fields: name + 21 parameters
        if (tokens.size() < 22) {
            throw std::runtime_error("Invalid CSV line: insufficient fields");
        }

        Camera camera;
        size_t idx = 0;

        try {
            // Parse name
            camera.name = tokens[idx++];

            // Parse extrinsics: translation (tx, ty, tz) as 3x1 matrix
            camera.extrinsics.translation = cv::Mat(3, 1, CV_64F);
            camera.extrinsics.translation.at<double>(0, 0) = std::stod(tokens[idx++]); // tx
            camera.extrinsics.translation.at<double>(1, 0) = std::stod(tokens[idx++]); // ty
            camera.extrinsics.translation.at<double>(2, 0) = std::stod(tokens[idx++]); // tz

            // Parse extrinsics: rotation matrix (row-major, 9 elements)
            camera.extrinsics.rotation = cv::Mat::zeros(3, 3, CV_64F);
            camera.extrinsics.rotation.at<double>(0, 0) = std::stod(tokens[idx++]); // R00
            camera.extrinsics.rotation.at<double>(0, 1) = std::stod(tokens[idx++]); // R01
            camera.extrinsics.rotation.at<double>(0, 2) = std::stod(tokens[idx++]); // R02
            camera.extrinsics.rotation.at<double>(1, 0) = std::stod(tokens[idx++]); // R10
            camera.extrinsics.rotation.at<double>(1, 1) = std::stod(tokens[idx++]); // R11
            camera.extrinsics.rotation.at<double>(1, 2) = std::stod(tokens[idx++]); // R12
            camera.extrinsics.rotation.at<double>(2, 0) = std::stod(tokens[idx++]); // R20
            camera.extrinsics.rotation.at<double>(2, 1) = std::stod(tokens[idx++]); // R21
            camera.extrinsics.rotation.at<double>(2, 2) = std::stod(tokens[idx++]); // R22

            // Parse intrinsics: camera matrix K = [[f, 0, px], [0, f, py], [0, 0, 1]]
            double f = std::stod(tokens[idx++]);      // f_pix
            double px = std::stod(tokens[idx++]);     // px_pix
            double py = std::stod(tokens[idx++]);     // py_pix

            camera.intrinsics.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            camera.intrinsics.camera_matrix.at<double>(0, 0) = f;
            camera.intrinsics.camera_matrix.at<double>(1, 1) = f;
            camera.intrinsics.camera_matrix.at<double>(0, 2) = px;
            camera.intrinsics.camera_matrix.at<double>(1, 2) = py;

            // Parse distortion coefficients (Brown model - OpenCV ordering: k1, k2, p1, p2, k3, k4)
            camera.intrinsics.distortion = cv::Mat(6, 1, CV_64F);
            camera.intrinsics.distortion.at<double>(0, 0) = std::stod(tokens[idx++]); // k1
            camera.intrinsics.distortion.at<double>(1, 0) = std::stod(tokens[idx++]); // k2
            camera.intrinsics.distortion.at<double>(2, 0) = std::stod(tokens[idx++]); // p2 (t2)
            camera.intrinsics.distortion.at<double>(3, 0) = std::stod(tokens[idx++]); // p1 (t1)
            camera.intrinsics.distortion.at<double>(4, 0) = std::stod(tokens[idx++]); // k3
            camera.intrinsics.distortion.at<double>(5, 0) = std::stod(tokens[idx++]); // k4

            cameras.push_back(camera);
        }
        catch (const std::invalid_argument& e) {
            throw std::runtime_error("Failed to parse camera parameters: invalid numeric value");
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Failed to parse camera parameters: numeric value out of range");
        }
    }

    if (cameras.empty()) {
        throw std::runtime_error("No valid camera entries found in CSV file");
    }

    return cameras;
}