#include "include/zncc_processor.h"
#include "include/image_handler.h"
#include "include/camera_loader.h"
#include "include/normal_permutation_generator.h"
#include "include/cloud_loader.h"
#include "include/io_utils.h"
#include "include/homography_utils.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <array>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <unordered_map>

// i should probably make some Params structures
const bool DBG_SINGLE_CAMERA = false;               // for debugging, only use 1 camera
const bool SHOW_PROPAGATION_PREVIEW = true;         // shows the propagation frontier as it grows
const bool SAVE_DEBUG_DATA = false;                 // save depth, normal, score maps as images and normal permutations as xyz
const bool REMOVE_SFM_POINTS = true;                // keep that on !!!
const float DEPTH_STEP_SIZE_PX = 0.25f;             // no more guessing, step size in refCam pixels
const float BEST_CAMERA_ANGLE_DEGREES = 17.5f;      // 17.5 is ok
const float NORMAL_DOT_THRESHOLD = 0.75f;           // dot product of normal and camera direction, used to filter out normals that are facing away from the camera
const float WINDOW_SCORE_THRESHOLD = 0.8f;          // this should be the same as the ZNCC threshold in the kernel
const float REFINE_DEPTH_STEP_MULTIPLIER = 0.5f;    // multiplier for depth step size during refinement, smaller = finer steps
const int WINDOW_RADIUS = 3;                        // 3-5 works well << if clouds come out noisy then this is too small, if clouds come out "oversmoothed" then this is too large
const int DEPTH_STEPS = 16;                         // originally 16
const int NORMAL_PERMUTATIONS = 16;                 // originally 16
const int REFINE_ITERATIONS = 1;                    // number of separate refinement steps
const int FRONTIER_MIN_SIZE = 100;                  // early stop except 1st frontier
const int MAX_ITERATIONS = 1000;                    // adjust for image size
const int TCAMS = 2;                                // not a parameter really, this is just the way the kernel is written

const std::vector<std::pair<float, int>> NORMAL_PERMUTATION_RINGS = { {3.0f, 5}, {6.0f, 5},  {12.0f, 5} };
const std::vector<std::pair<int, int>> PROPAGATION_OFFSETS = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} }; // propagation pattern

// 16 : {{4.f, 7}, {14.f, 8}}
// 16 : {{3.f, 4}, {8.f, 5}, {18.f, 6}}
// 32 : {{3.f, 15}, {12.f, 16}}
// 32 : {{2.5f, 8}, {7.f, 10}, {18.f, 13}}
// 32 : {{2.f, 6}, {5.f, 7}, {10.f, 8}, {20.f, 10}}
// 32 : {{2.f, 4}, {4.f, 5}, {8.f, 6}, {14.f, 7}, {24.f, 9}}

// 4 cross pattern offsets: {{0, -1}, {0, 1}, {-1, 0}, {1, 0}}
// 8 neighbor offsets: {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
// 12 neighbor offsets: {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}, {-2, 0}, {2, 0}, {0, -2}, {0, 2}}

// TODO:
// - add color to dense point clouds MEH
// - add more angles to normal permutations DONE
// - add normal vector dot product during propagation to filter out "impossible" normals DONE
// - try different zncc aggregation methods
// - switch to disparity?
// - find the pdf with "chessboard pattern propagation" or "red-black" (gipuma?) << https://openaccess.thecvf.com/content_iccv_2015/papers/Galliani_Massively_Parallel_Multiview_ICCV_2015_paper.pdf doesn't really solve anything
// - precompute normals in the sparse point cloud DONE-ish
// - add refine steps
// - auto calculate DEPTH_STEP_SIZE somehow

// move to utils
void reprojectViewToView(
    const Camera& srcCam,
    const cv::Mat& srcDepthMap,
    const cv::Mat& srcNormalMap,
    const cv::Mat& srcScoreMap,
    const Camera& dstCam,
    cv::Mat& dstDepthMap,
    cv::Mat& dstNormalMap,
    cv::Mat& dstScoreMap,
    const float scoreThreshold)
{
    const cv::Mat& srcR = srcCam.extrinsics.rotation;
    const cv::Mat& srcT = srcCam.extrinsics.translation;
    const cv::Mat& dstR = dstCam.extrinsics.rotation;
    const cv::Mat& dstT = dstCam.extrinsics.translation;
    const cv::Mat srcKinv = srcCam.intrinsics.camera_matrix.inv();
    const cv::Mat& dstK = dstCam.intrinsics.camera_matrix;
	
    const cv::Mat R_srcToDst = dstR * srcR.t(); // relative rotation from source camera to destination camera

    for (int r = 0; r < srcDepthMap.rows; ++r) {
        for (int c = 0; c < srcDepthMap.cols; ++c) {
            const float depth = srcDepthMap.at<float>(r, c);
            if (!std::isfinite(depth) || depth <= 0.f) continue;

            const float score = srcScoreMap.at<float>(r, c);
            if (!std::isfinite(score) || score < scoreThreshold) continue;

            // back projection
            const cv::Vec3d ray = pixelRay(srcKinv, c, r); 
            const cv::Mat Xsrc = (cv::Mat_<double>(3, 1) << ray[0] * depth, ray[1] * depth, ray[2] * depth);
            
            const cv::Mat Xworld = srcR.t() * (Xsrc - srcT); // w2c
            
            const cv::Mat Xdst = dstR * Xworld + dstT; 

            const double z = Xdst.at<double>(2, 0);
            if (!std::isfinite(z) || z <= 0.0) continue;

            const double u = dstK.at<double>(0, 0) * Xdst.at<double>(0, 0) / z + dstK.at<double>(0, 2); // project to dst
            const double v = dstK.at<double>(1, 1) * Xdst.at<double>(1, 0) / z + dstK.at<double>(1, 2);

            if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0 || u >= dstDepthMap.cols || v < 0.0 || v >= dstDepthMap.rows) continue;

            const int x = static_cast<int>(std::lround(u));
            const int y = static_cast<int>(std::lround(v));

            if (x < 0 || x >= dstDepthMap.cols || y < 0 || y >= dstDepthMap.rows) continue;

            float& dstDepth = dstDepthMap.at<float>(y, x);
            float& dstScore = dstScoreMap.at<float>(y, x);
            
			const bool destinationEmpty = dstDepth <= 0.f || !std::isfinite(dstDepth); // ignore invalid depths

			const bool replace = destinationEmpty || score > dstScore || (score == dstScore && z < dstDepth); // primitive z-buffering with score

            if (!replace) continue;
            
            const cv::Vec3f srcNormal = srcNormalMap.at<cv::Vec3f>(r, c);

            const cv::Mat normalSrc = (cv::Mat_<double>(3, 1) << srcNormal[0], srcNormal[1], srcNormal[2]);

            const cv::Mat normalDst = R_srcToDst * normalSrc;

            cv::Vec3f dstNormal(                                  // this could be shortened to -normalDst.at<double>(2, 0)
                static_cast<float>(normalDst.at<double>(0, 0)),
                static_cast<float>(normalDst.at<double>(1, 0)),
                static_cast<float>(normalDst.at<double>(2, 0)));

			// should check if the normal is facing the camera
            float dotProduct = dstNormal[0] * Xdst.at<double>(0, 0) +
                dstNormal[1] * Xdst.at<double>(1, 0) +
				dstNormal[2] * Xdst.at<double>(2, 0); 

            if (dotProduct < NORMAL_DOT_THRESHOLD) {
				continue; // skip normals that are facing away from the camera
			}   

            const float normalLength = cv::norm(dstNormal);
            if (normalLength > 1e-6f)
                dstNormal /= normalLength;

            dstDepth = static_cast<float>(z);
            dstScore = score;
            dstNormalMap.at<cv::Vec3f>(y, x) = dstNormal;
        }
    }
}

int main() {    

    std::string folderPath = "data\\buddaFull\\";
    std::vector<Camera> cameras;
    {
        CameraLoader cameraLoader;
        cameras = cameraLoader.loadFromCSV("data\\buddaFull\\budda.csv");
    }

    if (cameras.size() < 3) {
        std::cerr << "Error: At least 3 cameras are required for this algorithm." << std::endl;
        return -1;
	}

    std::unordered_map<int, std::unordered_map<int, float>> cameraAngle = calculateCameraAxisAngles(cameras);
    ImageHandler imageHandler;

    NormalPermutationGenerator normalPermGen(NORMAL_PERMUTATIONS);
    std::vector<NormalPermutation> normalPermutations = normalPermGen.generateMulti(NORMAL_PERMUTATION_RINGS);
	//std::vector<NormalPermutation> normalPermutations = { {0.0f, 0.0f} }; // only the parent normal, no permutations

    if (SAVE_DEBUG_DATA) {
        normalPermGen.saveNormalsAsXyz("dbg\\normals_debug.xyz", normalPermutations);
    }

	const int numCameras = DBG_SINGLE_CAMERA ? 1 : static_cast<int>(cameras.size());

    // for later merging
    std::unordered_map<int, cv::Mat> allDepthMaps;
    std::unordered_map<int, cv::Mat> allNormalMaps;
	std::unordered_map<int, cv::Mat> allScoreMaps;

    for (int refCamIdx = 0; refCamIdx < numCameras; ++refCamIdx) {

        auto bestCameras = findBestCamerasByAngleGaussian(refCamIdx, TCAMS, cameraAngle, BEST_CAMERA_ANGLE_DEGREES);

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

        CloudLoader cloudLoader;

        auto refDepthNormals = cloudLoader.loadXYZToDepthMapWithNormals("data\\buddaFull\\budda.xyz", cameras[refCamIdx], refImageGPU, true);

        const int IMG_WIDTH = refImageGPU.width;
        const int IMG_HEIGHT = refImageGPU.height;
        const double progressImageScale = 900.f / float(std::max(IMG_WIDTH, IMG_HEIGHT));

        cv::Mat scoreMapHost = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32FC1);

		int camsToReproject = std::min(static_cast<int>(cameraAngle[refCamIdx].size()), 2); // limit to 2 cameras for speed

        for (const auto& cam : cameraAngle[refCamIdx]) {
			if (allDepthMaps.find(cam.first) == allDepthMaps.end()) continue; // skip if no depth map available
			if (camsToReproject <= 0) break;
            int srcCamIdx = cam.first;
            if (srcCamIdx == refCamIdx) continue; // skip self
            reprojectViewToView(cameras[srcCamIdx], allDepthMaps[srcCamIdx], allNormalMaps[srcCamIdx], allScoreMaps[srcCamIdx],
                cameras[refCamIdx], refDepthNormals.depthMap, refDepthNormals.normalMap, scoreMapHost, WINDOW_SCORE_THRESHOLD);
			camsToReproject--;
		}        

        const float medianDepth = cloudLoader.getMedianDepth();
		const float calculatedDepthStepSizeTarget0 = calcDepthStepForPixelShift(cameras[refCamIdx], cameras[targetIdx0], medianDepth, cv::Point2d(IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0), DEPTH_STEP_SIZE_PX);
		const float calculatedDepthStepSizeTarget1 = calcDepthStepForPixelShift(cameras[refCamIdx], cameras[targetIdx1], medianDepth, cv::Point2d(IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0), DEPTH_STEP_SIZE_PX);

		float calculatedDepthStepSize = std::min(calculatedDepthStepSizeTarget0, calculatedDepthStepSizeTarget1); // maybe avg ?

		std::cout << "Calculated depth step size: " << calculatedDepthStepSize << std::endl;        

        ZNCCProcessor processor(IMG_WIDTH, IMG_HEIGHT, WINDOW_RADIUS);

        std::vector<float> depthSteps = generateDepthSteps(calculatedDepthStepSize, DEPTH_STEPS);

        HomographyParams hp0 = calcHomographyParamsW2C(cameras[refCamIdx], cameras[targetIdx0]);
        HomographyParams hp1 = calcHomographyParamsW2C(cameras[refCamIdx], cameras[targetIdx1]);        

        processor.setCameraMatrices(hp0.H0.data(), hp0.K1inv.data(), hp0.e2.data());
        processor.setCameraMatricesSecond(hp1.H0.data(), hp1.K1inv.data(), hp1.e2.data());
        processor.setNormalPermutations(normalPermutations.data(), static_cast<int>(normalPermutations.size()));        

        // refine begin

        size_t refineStep = 0;                

        std::vector<FrontierPoint> frontierPoints;

		const int nonZeroCount = cv::countNonZero(refDepthNormals.depthMap > 0.f);

        frontierPoints.reserve(nonZeroCount);
        
        // initialize with smf points
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
                    //fp.normal = make_float3(initialNormal[0], initialNormal[1], initialNormal[2]);
                    fp.normal = make_float3(normal[0], normal[1], normal[2]);
                    frontierPoints.push_back(fp);
					scoreMapHost.at<float>(y, x) = WINDOW_SCORE_THRESHOLD; // should be equal to score threshold in kernel
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
					scoreMapHost.at<float>(rp.y, rp.x) = rp.znccScore;
                }
            }

            // build next frontier
            std::vector<FrontierPoint> nextFrontier;
            nextFrontier.reserve(results.size() * 4);
            for (const auto& rp : results) {

                float parentNormalCamDot = rp.normal.z * -1.f; // cam dir is (0,0,-1) so dot can be shortened
                if (parentNormalCamDot < NORMAL_DOT_THRESHOLD) // keep only normals that are facing the camera at max ~60deg 
                    continue;

                for (const auto& offset : PROPAGATION_OFFSETS) { // cannot use auto& [dx, dy] because of nvcc
                    if (offset.first == 0 && offset.second == 0) continue; // thats not neede anymore
                    int nx = rp.x + offset.first;
                    int ny = rp.y + offset.second;
                    if (nx < 0 || nx >= IMG_WIDTH || ny < 0 || ny >= IMG_HEIGHT) continue;                    
					if (scoreMapHost.at<float>(ny, nx) <= 0.0f) { // only propagate to uninitialized pixels
                        FrontierPoint child;
                        child.x = nx;
                        child.y = ny;
                        child.depth = rp.depth;
                        child.normal = rp.normal;
                        nextFrontier.push_back(child);                        
                        scoreMapHost.at<float>(ny, nx) = WINDOW_SCORE_THRESHOLD; // propagate the base score
                    }
                }
            }

            frontierPoints.swap(nextFrontier);

            if (SHOW_PROPAGATION_PREVIEW) { // i guess this isn't good for performance but the whole program is in debug state anyway
                cv::Mat progressImage;
                cv::resize(refDepthNormals.depthMap, progressImage, cv::Size(), progressImageScale, progressImageScale, cv::INTER_NEAREST);
                progressImage.convertTo(progressImage, CV_8U, 255.f, 0.0f);
                cv::imshow("Depth Map Progress", progressImage);
                cv::waitKey(1);
            }
        }

        for (int refineIter = 0; refineIter < REFINE_ITERATIONS; ++refineIter) {
            calculatedDepthStepSize *= REFINE_DEPTH_STEP_MULTIPLIER;
            auto refineDepthSteps = generateDepthSteps(calculatedDepthStepSize, DEPTH_STEPS);
            std::cout << "Refinement Iteration " << refineIter + 1 << ", new depth step size: " << calculatedDepthStepSize << std::endl;

			// build a single frontier from all valid depth pixels
			frontierPoints.clear();
			int refineNonZeroCount = cv::countNonZero(refDepthNormals.depthMap > 0.f);
			frontierPoints.reserve(refineNonZeroCount);

            for (int y = 0; y < IMG_HEIGHT; ++y) {
                for (int x = 0; x < IMG_WIDTH; ++x) {
                    float d = refDepthNormals.depthMap.at<float>(y, x);
                    if (d > 0.f) {
                        cv::Vec3f normal = refDepthNormals.normalMap.at<cv::Vec3f>(y, x);
                        FrontierPoint fp;
                        fp.x = x;
                        fp.y = y;
                        fp.depth = d;
                        fp.normal = make_float3(normal[0], normal[1], normal[2]);
                        frontierPoints.push_back(fp);
                    }
                }
			}

            std::vector<ResultPoint> results = processor.computeZNCCDualTargets(
                refImageGPU,
                targetImage0GPU,
                targetImage1GPU,
                refDepthNormals.depthMap,
                frontierPoints,
                refineDepthSteps,
                static_cast<int>(normalPermutations.size()));

			// return points with better score and update depth map, normal map and visited mask
            for (const auto& rp : results) {
                if (rp.y >= 0 && rp.y < IMG_HEIGHT && rp.x >= 0 && rp.x < IMG_WIDTH) {
                    float& currentScore = scoreMapHost.at<float>(rp.y, rp.x);
                    if (rp.znccScore > currentScore) { // only update if the new score is better
                        refDepthNormals.depthMap.at<float>(rp.y, rp.x) = rp.depth;
                        refDepthNormals.normalMap.at<cv::Vec3f>(rp.y, rp.x) = cv::Vec3f(rp.normal.x, rp.normal.y, rp.normal.z);
                        currentScore = rp.znccScore;
                    }
                }
			}
		}

        cv::destroyAllWindows();

        // refine end

        std::string pointCloudFilename = "clouds\\cloud_" + std::to_string(refCamIdx) + "_w" + std::to_string(1 + WINDOW_RADIUS * 2) + "_c" + std::to_string(8) + "_dss" + std::to_string(calculatedDepthStepSize) + ".xyz";
        saveCloudsAsXyz(pointCloudFilename, refDepthNormals.normalMap, refDepthNormals.depthMap, cameras[refCamIdx], true);

        if (SAVE_DEBUG_DATA) {
            saveNormalMapPng("dbg\\result_normal_map_" + std::to_string(refCamIdx) + ".png", refDepthNormals.normalMap, refDepthNormals.depthMap);
            cv::Mat depthMapToSave, depthMapNormalized;
            cv::normalize(refDepthNormals.depthMap, depthMapNormalized, 0, 1, cv::NORM_MINMAX);
            depthMapNormalized.convertTo(depthMapToSave, CV_8U, 255.0f);
            cv::imwrite("dbg\\final_depth_map_" + std::to_string(refCamIdx) + ".png", depthMapToSave);
			cv::Mat scoreMapToSave, scoreMapNormalized;
			cv::normalize(scoreMapHost, scoreMapNormalized, 0, 1, cv::NORM_MINMAX);
			scoreMapNormalized.convertTo(scoreMapToSave, CV_8U, 255.0f);
			cv::imwrite("dbg\\final_score_map_" + std::to_string(refCamIdx) + ".png", scoreMapToSave);
        }

        imageHandler.freeGPUImage(refImageGPU);
        imageHandler.freeGPUImage(targetImage0GPU);
        imageHandler.freeGPUImage(targetImage1GPU);

		allDepthMaps[refCamIdx] = refDepthNormals.depthMap.clone();
		allNormalMaps[refCamIdx] = refDepthNormals.normalMap.clone();
		allScoreMaps[refCamIdx] = scoreMapHost.clone();

        std::cout << "\n\nReference Camera[" << refCamIdx << "] Iterations: " << iteration << "\n";
    }
    
    return 0;
}
