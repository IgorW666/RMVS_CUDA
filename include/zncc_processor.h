#pragma once

#include "image_handler.h"
#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <utility> // for std::pair

#include <opencv2/opencv.hpp>

// yeah, so some design flaws here: this wrapper separates CUDA code from "harmful" libraries like openCV
// but i need a depth map available here. Passing a device pointer doesn't solve the problem because
// I cannot read a gpu pointer on the host

/*
* zncc Score will have pitch value according to the number of tested depth and normal hypotheses
* for every depth hypothesis, there will be a number of normal hypotheses tested
* so the step of output will have to be decoded properly to get the correct score for a given depth and normal hypothesis
* everything has to fit to a block nicely so the numbers are critical
* probable first try for 256 threads per block:
*
* 16 depth hypotheses
*   each over 16 normal hypotheses
* so the array will be: d0n0, d0n1, d0n2, ..., d0n15, d1n0, d1n1, ..., d15n15
* so to get the score for depth hypothesis d and normal hypothesis n, the index will be: d * normalSteps + n
*/
struct ZNCCResult {
    float* gpuData;
    int width;
    int height;
    int(depthSteps);     // Number of depth hypotheses
    int(normalSteps);    // Number of normal hypotheses
    size_t pitch;       // Pitch for GPU memory (bytes)
    //ZNCCResult() : gpuData(nullptr), width(0), height(0), pitch(0) {}
};

class ZNCCProcessor {
public:
    ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius);
    ~ZNCCProcessor();

    /**
    * Set camera data.
    *
    * @param H0    3x3 matrix K2 * R * K1^-1 (row-major)
    * @param K1inv 3x3 matrix K1^-1 (row-major)
    * @param e2    3-vector K2 * t
    */
    void setCameraMatrices(const float* H0, const float* K1inv, const float* e2);

    /**
    * Set second target camera data (non-breaking addition).
    *
    * @param H1    3x3 matrix for second target (row-major)
    * @param K1inv1 3x3 K1^-1 or K1inv for second target projection
    * @param e2_1  3-vector K2' * t'
    */
    void setCameraMatricesSecond(const float* H1, const float* K1inv1, const float* e2_1);


    void setNormalPermutations(const NormalPermutation* normalPermutations, int count); // normal permutations are small >> __constant__

   
    /**
     * Execute ZNCC computation for two target images in a single kernel pass.
     *
     * Returns a vector of consensus results where both targets agree on depth & normal hypothesis.
     */
    std::vector<ResultPoint> computeZNCCDualTargets(
        const ImageData& refImage,
        const ImageData& targetImage0,
        const ImageData& targetImage1,
		//const ImageData& depthMap, // for now it can be const but later it will be updated iside the kernel
		//const ImageData& normalMap, // for now it can be const but later it will be updated iside the kernel
		cv::Mat& depthMap, // hmm       
        const std::vector<FrontierPoint>& frontierPoints,
        const std::vector<float>& depthSteps,
        int normalNSteps);

private:
    int imgWidth;
    int imgHeight;
    int windowRadius;
    ImageData znccOutput;
    cv::Mat test;
	std::vector<NormalPermutation> normalPermutationsHost; // need that to reconstruct normals on host after kernel run
};