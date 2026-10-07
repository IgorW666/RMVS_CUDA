#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <iomanip>
#include <sstream>
#include <fstream>

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

const int WINDOW_RADIUS = 4;
const float DEPTH_MIN = 0.f;
const float DEPTH_MAX = 5.0f;
const int DEPTH_STEPS = 10000;

int main() {
    try {

		std::vector<Camera> cameras;

        {
            CameraLoader cameraLoader;
			cameras = cameraLoader.loadFromCSV("data\\budda\\budda.csv");
        }

        ImageHandler imageHandler;
        
        std::cout << "Loading images...\n";
        ImageData refImage = imageHandler.loadImageToGPU("data\\budda\\img\\41454.jpg"); // idx 0
        ImageData targetImage = imageHandler.loadImageToGPU("data\\budda\\img\\41457.jpg"); // idx 6

        const int IMG_WIDTH = refImage.width;
        const int IMG_HEIGHT = refImage.height;        

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);		

        HomographyParams hp = calcHomographyParams(cameras[0], cameras[3]);
        processor.setCameraMatrices(hp.H0.data(), hp.K1inv.data(), hp.e2.data());

        ImageData result = processor.computeZNCC(refImage, targetImage, depthMap, normalMap);
        
        cv::Mat resultHost = imageHandler.getImageFromGPU(result);

        imageHandler.freeGPUImage(depthMap);              
        imageHandler.freeGPUImage(normalMap);
        imageHandler.freeGPUImage(refImage);
        imageHandler.freeGPUImage(targetImage);

        std::cout << "Done!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}