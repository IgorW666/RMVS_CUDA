#pragma once

#include "image_handler.h"
#include "types.h"
#include <cuda_runtime.h>


/*
* Zncc Score will have pitch value according to the number of tested depth and normal hypotheses
* For every depth hypothesis, there will be a number of normal hypotheses tested
* So the step of output will have to be decoded properly to get the correct score for a given depth and normal hypothesis
* Everything has to fit to a block nicely so the numbers are critical
* Probable first try for 256 threads per block:
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
	int depthSteps;     // Number of depth hypotheses
	int normalSteps;    // Number of normal hypotheses
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


	void setNormalPermutations(const NormalPermutation* normalPermutations, int count); // normal permutations are small >> __constant__

    /**
     * Execute ZNCC computation.
     *
     * @param refImage      Reference image (GPU memory)
     * @param targetImage   Target image (GPU memory)
     * @param depthMap      Depth map (GPU memory)
     * @param normalMap     Normal map (GPU memory)
     * @return              Result ZNCC scores (GPU memory)
     */
    std::vector<ResultPoint> computeZNCC(
        const ImageData& refImage,
        const ImageData& targetImage,
        const std::vector<FrontierPoint>& frontierPoints,
        const std::vector<float>& depthSteps,
        int normalNSteps/*, const NormalPermutation* normalPermutations, const ImageData& depthMap, const ImageData& normalMap*/ );
    
private:
    int imgWidth;
    int imgHeight;
    int windowRadius;
    ImageData znccOutput;
};