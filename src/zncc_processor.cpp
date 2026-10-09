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

ImageData ZNCCProcessor::computeZNCC(
    const ImageData& refImage,
    const ImageData& targetImage,
    const std::vector<FrontierPoint>& frontierPoints,
    const std::vector<float>& depthSteps,
    int normalNSteps/*, const NormalPermutation* normalPermutations, const ImageData& depthMap, const ImageData& normalMap*/ ) {

    //const ImageData& depthMap, const ImageData& normalMap) {
    // Allocate output with pitch for consistency
    //if (!znccOutput.gpuData) { cudaMallocPitch(&znccOutput.gpuData, &znccOutput.pitch, imgWidth * sizeof(float), imgHeight); }       

    // we need:
	// FrontierPoint* frontier // that should probably be allocated to GPU here and passed to the kernel
    // float* depthSteps       // not really sure about this yet
	// float* result
	// NormalPermutation * normalSteps

    const int frontierCount = static_cast<int>(frontierPoints.size());

    FrontierPoint* frontierPtr = nullptr;
	cudaMalloc((void**)&frontierPtr, frontierCount * sizeof(FrontierPoint)); // contiguous
    cudaMemcpy(frontierPtr, frontierPoints.data(), frontierCount * sizeof(FrontierPoint), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GPU copy failed: ") + cudaGetErrorString(err));
    }

	float* depthStepsPtr = nullptr;
	const int depthCount = static_cast<int>(depthSteps.size());
	cudaMalloc((void**)&depthStepsPtr, depthCount * sizeof(float));
	cudaMemcpy(depthStepsPtr, depthSteps.data(), depthCount * sizeof(float), cudaMemcpyHostToDevice);

	// compute expected result size (floats)
    const int windowSize = 1 + 2 * windowRadius;
	const int resultFloatsPerFrontier = depthCount * normalNSteps * windowSize * windowSize * 2;
	const int resultFloatsTotal = frontierCount * resultFloatsPerFrontier;

	float* result = nullptr;
	cudaMalloc((void**)&result, resultFloatsTotal * sizeof(float));
    
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
        frontierCount); // <-- pass frontier count

    checkCudaError("computeZNCC kernel launch");

	cv::Mat resultMat(targetImage.height, targetImage.width, CV_32F, cv::Scalar(0.f));

	// Copy result back to host (allocate host vector)
	std::vector<float> resultHost;
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

    int resultUVs = resultFloatsTotal / 2;

	int validCount = 0;

    for (int i = 0; i < resultUVs; ++i) {
		float uf = resultHost[i * 2];
		float vf = resultHost[i * 2 + 1];

		// invalid marker from kernel is -1.f
        if (uf < 0.f || vf < 0.f || uf >= static_cast<float>(targetImage.width) || vf >= static_cast<float>(targetImage.height)) {
			// skip invalid coordinates
            continue;
		}

		int u = static_cast<int>(std::round(uf));
		int v = static_cast<int>(std::round(vf));

        if (u < 0 || u >= targetImage.width || v < 0 || v >= targetImage.height) {
            continue;
        }

		resultMat.at<float>(v, u) = 1.f;
		validCount++;
	}

	std::cout << "Total valid count: " << validCount << ", ~NWindows : " << validCount / (windowSize * windowSize) << std::endl;

    // Upload resultMat to GPU and return as ImageData (znccOutput)
    size_t bytes = static_cast<size_t>(targetImage.width) * static_cast<size_t>(targetImage.height) * sizeof(float);

    // free previous gpuData if any
    if (znccOutput.gpuData) {
        cudaFree(znccOutput.gpuData);
        znccOutput.gpuData = nullptr;
    }

    cudaMalloc((void**)&znccOutput.gpuData, bytes);
    cudaMemcpy(znccOutput.gpuData, resultMat.ptr<float>(0), bytes, cudaMemcpyHostToDevice);
    znccOutput.width = targetImage.width;
    znccOutput.height = targetImage.height;
    znccOutput.pitch = 0; // not using pitched allocation here

    // cleanup temporaries
    cudaFree(frontierPtr);
    cudaFree(depthStepsPtr);
    cudaFree(result);

    return znccOutput;
}