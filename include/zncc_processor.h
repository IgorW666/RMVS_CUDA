#pragma once

#include "image_handler.h"
#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <utility> // for std::pair

#include <opencv2/opencv.hpp>

class ZNCCProcessor {
public:
    ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius);
    ~ZNCCProcessor();

    void setCameraMatrices(const float* H0_0, const float* K1inv_0, const float* e2_0);

    void setCameraMatricesSecond(const float* H0_1, const float* K1inv_1, const float* e2_1);

    void setNormalPermutations(const NormalPermutation* normalPermutations, int count);

    std::vector<ResultPoint> computeZNCCDualTargets(
        const ImageData& refImage,
        const ImageData& targetImage0,
        const ImageData& targetImage1,		
		cv::Mat& depthMap, // hmm       
        const std::vector<FrontierPoint>& frontierPoints,
        const std::vector<float>& depthSteps,
        int normalNSteps);

private:
    int imgWidth;
    int imgHeight;
    int windowRadius;
    //ImageData znccOutput;
    //cv::Mat test;
	std::vector<NormalPermutation> normalPermutationsHost; // need that to reconstruct normals on host after kernel run
};