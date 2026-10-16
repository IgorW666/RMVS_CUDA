#include "../include/zncc_processor.h"
#include "../include/cuda_homography.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath> // for std::round, sqrtf
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <utility>

constexpr float znccThreshold = 0.7f; // threshold for ZNCC score to consider a hypothesis valid (for all score actually)
//constexpr bool induceDepthDevPenalty = false; // whether to apply depth deviation penalty in ZNCC evaluation << whooa, keep this off!! Results are alot smoother without it !!!
//constexpr float depthDevPenaltyStrength = 1.0f; // penalty factor for depth deviation (1 = no change) , too much and the clouds will be "stepped"

constexpr int localDepthWindowOffset = 3; // offset for stddev calculation, 0 = center of window, 1 = one pixel to the right, etc. (for testing only)

ZNCCProcessor::ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius)
    : imgWidth(imgWidth), imgHeight(imgHeight), windowRadius(windowRadius) {
    setWindowRadius(windowRadius);
    znccOutput.gpuData = nullptr;
    znccOutput.width = imgWidth;
    znccOutput.height = imgHeight;
    znccOutput.pitch = 0;
}

ZNCCProcessor::~ZNCCProcessor() {
    if (znccOutput.gpuData) cudaFree(znccOutput.gpuData);
}

void ZNCCProcessor::setCameraMatrices(const float* H0, const float* K1inv, const float* e2) {
    setupCameraMatrices(H0, K1inv, e2);
}

void ZNCCProcessor::setCameraMatricesSecond(const float* H1, const float* K1inv1, const float* e2_1) {    
    setupCameraMatricesSecond(H1, K1inv1, e2_1);
}

void ZNCCProcessor::setNormalPermutations(const NormalPermutation* normalPermutations, int count) {    
    normalPermutationsHost.assign(normalPermutations, normalPermutations + count); // keep a host-side copy so we can reconstruct hypothesis normals after the kernel run
    setupNormalPermutations(normalPermutations, count);
}

// host helpers (simple vector math used to reconstruct normals on host)
static inline float3 host_normalize(const float3& v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 1e-8f) return make_float3(0.0f, 0.0f, 1.0f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

static inline float3 host_cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

std::vector<ResultPoint> ZNCCProcessor::computeZNCCDualTargets(
    const ImageData& refImage,
    const ImageData& targetImage0,
    const ImageData& targetImage1,    
	cv::Mat& depthMap,
    const std::vector<FrontierPoint>& frontierPoints,
    const std::vector<float>& depthSteps,
    int normalNSteps) {

    const int frontierCount = static_cast<int>(frontierPoints.size());
    if (frontierCount == 0 || depthSteps.empty() || normalNSteps <= 0) {
        return {};
    }

    FrontierPoint* frontierPtr = nullptr;
    float* depthStepsPtr = nullptr;
    float* resultDev0 = nullptr;
    float* resultDev1 = nullptr;

    const int depthCount = static_cast<int>(depthSteps.size());
    const int resultFloatsPerFrontier = depthCount * normalNSteps;
    const int resultFloatsTotal = frontierCount * resultFloatsPerFrontier;

    try {
        // frontier array
        if (cudaMalloc((void**)&frontierPtr, frontierCount * sizeof(FrontierPoint)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc frontierPtr failed");
        if (cudaMemcpy(frontierPtr, frontierPoints.data(), frontierCount * sizeof(FrontierPoint), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy frontierPtr failed");

        // depth steps array
        if (cudaMalloc((void**)&depthStepsPtr, depthCount * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc depthStepsPtr failed");
        if (cudaMemcpy(depthStepsPtr, depthSteps.data(), depthCount * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy depthStepsPtr failed");

        // set three image pitches for device constant memory
        setImagePitchesDual(refImage.pitch, targetImage0.pitch, targetImage1.pitch);

        // result buffers on device
        if (cudaMalloc((void**)&resultDev0, resultFloatsTotal * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc resultDev0 failed");
        if (cudaMalloc((void**)&resultDev1, resultFloatsTotal * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc resultDev1 failed");

        // launch dual-target kernel
        launchEvaluateZNCC_DualTargets(
            refImage.gpuData,
            targetImage0.gpuData,
            targetImage1.gpuData,
            frontierPtr,
            depthStepsPtr,
            resultDev0,
            resultDev1,
            depthCount,
            normalNSteps,
            refImage.width,
            refImage.height,
            targetImage0.width,
            targetImage0.height,
            targetImage1.width,
            targetImage1.height,
            frontierCount);

        checkCudaError("computeZNCCDualTargets kernel launch");

        // copy results back to host
        std::vector<float> resultHost0(resultFloatsTotal);
        std::vector<float> resultHost1(resultFloatsTotal);
        if (cudaMemcpy(resultHost0.data(), resultDev0, resultFloatsTotal * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy resultDev0 DeviceToHost failed");
        if (cudaMemcpy(resultHost1.data(), resultDev1, resultFloatsTotal * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy resultDev1 DeviceToHost failed");

        std::vector<ResultPoint> consensusResults;
        consensusResults.reserve(frontierCount);

        auto reconstruct_normal = [&](int fIdx, int normalIdx) -> float3 {       // declutter this into a standalone function
            if (normalIdx >= 0 && normalIdx < static_cast<int>(normalPermutationsHost.size())) {
                NormalPermutation perm = normalPermutationsHost[normalIdx];
                float3 parentNormal = host_normalize(frontierPoints[fIdx].normal);
                float3 reference;
                if (std::fabs(parentNormal.z) < 0.999f) reference = make_float3(0.0f, 0.0f, 1.0f);
                else reference = make_float3(0.0f, 1.0f, 0.0f);
                float3 parentTangent = host_normalize(host_cross(reference, parentNormal));
                float3 parentBitangent = host_normalize(host_cross(parentNormal, parentTangent));
                float3 hypothesis = make_float3(
                    parentNormal.x + perm.tangent * parentTangent.x + perm.bitangent * parentBitangent.x,
                    parentNormal.y + perm.tangent * parentTangent.y + perm.bitangent * parentBitangent.y,
                    parentNormal.z + perm.tangent * parentTangent.z + perm.bitangent * parentBitangent.z
                );
                return host_normalize(hypothesis);
            }
            else {
                return host_normalize(frontierPoints[fIdx].normal);
            }
            };


        for (int f = 0; f < frontierCount; ++f) {
            float bestJointScore = -std::numeric_limits<float>::infinity();
            int bestDepthIdx = 0;
            int bestNormalIdx = 0;

            const int baseIdx = f * resultFloatsPerFrontier;

			int localDepthCount = 0;			
			float localDepthMean = 0.f;

            for (int offsetx = -localDepthWindowOffset; offsetx <= localDepthWindowOffset; ++offsetx) {
                for (int offsety = -localDepthWindowOffset; offsety <= localDepthWindowOffset; ++offsety) {
					if (offsetx == 0 && offsety == 0) continue; // skip the center point
                    int neighborX = frontierPoints[f].x + offsetx;
                    int neighborY = frontierPoints[f].y + offsety;
                    if (neighborX >= 0 && neighborX < depthMap.cols && neighborY >= 0 && neighborY < depthMap.rows) {
                        float neighborDepth = depthMap.at<float>(neighborY, neighborX);
                        if (std::isfinite(neighborDepth) && neighborDepth > 0.f) {							
							localDepthMean += neighborDepth;
							localDepthCount++;
                        }
                    }
                }
			}

			localDepthMean /= (localDepthCount > 0) ? static_cast<float>(localDepthCount) : frontierPoints[f].depth; // avoid div by zero

            if (localDepthCount > 0) {
                bool wait1 = 1;
            }            

            for (int d = 0; d < depthCount; ++d) {
                const int depthBase = baseIdx + d * normalNSteps;
				const float depthHypothesis = frontierPoints[f].depth + depthSteps[d];
				const float depthDeviation = (localDepthCount > 0) ? std::fabs(depthHypothesis - localDepthMean) : 0.f;
                bool wait2 = 1;
                for (int n = 0; n < normalNSteps; ++n) {
                    const float score0 = resultHost0[depthBase + n];
                    const float score1 = resultHost1[depthBase + n];

                    if (!std::isfinite(score0) || !std::isfinite(score1))
                        continue;

                    float jointScore = 0.5f * (score0 + score1);

					jointScore -= depthDeviation * 1.0f;

                    if (jointScore > bestJointScore) {
                        bestJointScore = jointScore;
                        bestDepthIdx = d;
                        bestNormalIdx = n;
                    }
                }
            }

            if (!std::isfinite(bestJointScore))
                continue;

            if (bestJointScore < znccThreshold)
                continue;

            ResultPoint rp;
            rp.x = frontierPoints[f].x;
            rp.y = frontierPoints[f].y;
            rp.depth = frontierPoints[f].depth + depthSteps[bestDepthIdx];
            rp.znccScore = bestJointScore;
            rp.normal = reconstruct_normal(f, bestNormalIdx);

            consensusResults.push_back(rp);
        }

        cudaFree(frontierPtr);
        cudaFree(depthStepsPtr);
        cudaFree(resultDev0);
        cudaFree(resultDev1);

        return consensusResults;
    }
    catch (...) {
        if (frontierPtr) cudaFree(frontierPtr);
        if (depthStepsPtr) cudaFree(depthStepsPtr);
        if (resultDev0) cudaFree(resultDev0);
        if (resultDev1) cudaFree(resultDev1);
        throw;
    }
}



