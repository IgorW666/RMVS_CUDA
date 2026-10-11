#include "../include/zncc_processor.h"
#include "../include/cuda_homography.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath> // for std::round
#include <vector>

ZNCCProcessor::ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius)
    : imgWidth(imgWidth), imgHeight(imgHeight), windowRadius(windowRadius) {
    setWindowRadius(windowRadius);
    znccOutput.gpuData = nullptr;
    znccOutput.width = imgWidth;
    znccOutput.height = imgHeight;
    znccOutput.pitch = 0;
}

ZNCCProcessor::~ZNCCProcessor() {
    if (znccOutput.gpuData) {
        cudaFree(znccOutput.gpuData);
    }
}

void ZNCCProcessor::setCameraMatrices(const float* H0, const float* K1inv, const float* e2) {
    setupCameraMatrices(H0, K1inv, e2);
}

void ZNCCProcessor::setNormalPermutations(const NormalPermutation* normalPermutations, int count) {
    setupNormalPermutations(normalPermutations, count);
}

std::vector<ResultPoint> ZNCCProcessor::computeZNCC(
    const ImageData& refImage,
    const ImageData& targetImage,
    const std::vector<FrontierPoint>& frontierPoints,
    const std::vector<float>& depthSteps,
    int normalNSteps) {

    const int frontierCount = static_cast<int>(frontierPoints.size());
    
    // add a helper for that in the future
    FrontierPoint* frontierPtr = nullptr;
    cudaMalloc((void**)&frontierPtr, frontierCount * sizeof(FrontierPoint)); // contiguous
    cudaMemcpy(frontierPtr, frontierPoints.data(), frontierCount * sizeof(FrontierPoint), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GPU copy failed: ") + cudaGetErrorString(err));
    }

    // add a helper for that in the future
    float* depthStepsPtr = nullptr;
    const int depthCount = static_cast<int>(depthSteps.size());
    cudaMalloc((void**)&depthStepsPtr, depthCount * sizeof(float));
    cudaMemcpy(depthStepsPtr, depthSteps.data(), depthCount * sizeof(float), cudaMemcpyHostToDevice);
    
    setImagePitches(refImage.pitch, targetImage.pitch);

	// the results changed. Now we have a zncc score for depth/normal permutation for each frontier point
	// So the result array will be of size (frontierCount * depthCount * normmalNSteps)    
	const int resultFloatsPerFrontier = depthCount * normalNSteps;
	const int resultFloatsTotal = frontierCount * resultFloatsPerFrontier;	

	float* result = nullptr;
	cudaMalloc((void**)&result, resultFloatsTotal * sizeof(float)); // would be cool to initialize to -1.f
    
    // Call wrapper function with separate dimensions for ref and target images
    launchEvaluateZNCC(
        refImage.gpuData,
        targetImage.gpuData,
        frontierPtr,
		depthStepsPtr,           
        result,
        depthCount,
		normalNSteps,        // NormalPermutation array lives in __constant__ so I guess we need the number of steps here
        refImage.width,
        refImage.height,
        targetImage.width,
		targetImage.height,
        frontierCount);

    checkCudaError("computeZNCC kernel launch");

	//cv::Mat resultMat(targetImage.height, targetImage.width, CV_32F, cv::Scalar(0.f));
    //cv::Mat resultMatx4(targetImage.height*4, targetImage.width*4, CV_32F, cv::Scalar(0.f));

	// Copy result back to host (allocate host vector)
	std::vector<float> resultHost;
	std::vector<ResultPoint> validResults;
	
	resultHost.resize(resultFloatsTotal);
	cudaMemcpy(resultHost.data(), result, resultFloatsTotal * sizeof(float), cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
	    // cleanup before throwing
	    cudaFree(frontierPtr);
	    cudaFree(depthStepsPtr);
	    cudaFree(result);
	    throw std::runtime_error(std::string("cudaMemcpy DeviceToHost failed: ") + cudaGetErrorString(err));
	}

    //int resultUVs = resultFloatsTotal / 2;

	int validCount = 0;

    /*for (int i = 0; i < resultUVs; ++i) {
		float xf = resultHost[i * 2];
		float yf = resultHost[i * 2 + 1];

		float yfx4 = yf * 4.f;
		float xfx4 = xf * 4.f;

		// invalid marker from kernel is -1.f
        if (xf < 0.f || yf < 0.f || xf >= static_cast<float>(targetImage.width) || yf >= static_cast<float>(targetImage.height)) {
			// skip invalid coordinates
            continue;
		}

		int x = static_cast<int>(std::round(xf));
		int y = static_cast<int>(std::round(yf));

        if (x < 0 || x >= targetImage.width || y < 0 || y >= targetImage.height) {
            continue;
        }

        const int pairsPerWindow = windowSize * windowSize;                  // W*W
        const int pairsPerNormal = pairsPerWindow;                          // W*W
        const int pairsPerDepth = normalNSteps * pairsPerNormal;            // N * W*W
        const int pairsPerFrontier = depthCount * pairsPerDepth;            // D * N * W*W

        const int frontierIdx = i / pairsPerFrontier;
        const int withinFrontier = i % pairsPerFrontier;

        int depthHypothesisIdx = withinFrontier / pairsPerDepth;
        int normalHypothesisIdx = (withinFrontier / pairsPerNormal) % normalNSteps;

		//std::cout << "Valid coordinate : (" << xf << ", " << yf << ") at depth hypothesis index : " << depthHypothesisIdx << ", normal hypothesis index : " << normalHypothesisIdx << std::endl;

		resultMat.at<float>(y, x) = 1.f;
		resultMatx4.at<float>(static_cast<int>(std::round(yfx4)), static_cast<int>(std::round(xfx4))) = 1.f;
		validCount++;
	}*/

	//std::cout << "There should be " << resultUVs << " valid coordinates\n";
	//std::cout << "Total valid count: " << validCount << ", ~NWindows : " << validCount / (windowSize * windowSize) << std::endl;

    // Upload resultMat to GPU and return as ImageData (znccOutput)
    size_t bytes = static_cast<size_t>(targetImage.width) * static_cast<size_t>(targetImage.height) * sizeof(float);

    // free previous gpuData if any << not using ImageData as output anymore
    //if (znccOutput.gpuData) {
    //    cudaFree(znccOutput.gpuData);
    //    znccOutput.gpuData = nullptr;
    //}

    //cudaMalloc((void**)&znccOutput.gpuData, bytes);
    //cudaMemcpy(znccOutput.gpuData, resultMat.ptr<float>(0), bytes, cudaMemcpyHostToDevice);
    //znccOutput.width = targetImage.width;
    //znccOutput.height = targetImage.height;
    //znccOutput.pitch = 0; // not using pitched allocation here

    // cleanup temporaries
    cudaFree(frontierPtr);
    cudaFree(depthStepsPtr);
    cudaFree(result);

    return validResults;
}