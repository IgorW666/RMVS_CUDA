#include "../include/zncc_processor.h"
#include "../include/cuda_homography.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <stdexcept>
#include <utility>

constexpr float znccThreshold = 0.8f; // threshold for ZNCC score to consider a hypothesis valid (for all score actually) << this shoulbe passed as parameter in the constructor
constexpr int localDepthWindowOffset = 3; 

ZNCCProcessor::ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius)
    : imgWidth(imgWidth), imgHeight(imgHeight), windowRadius(windowRadius) {
    setWindowRadius(windowRadius);
    // all constant set functions go here
}

ZNCCProcessor::~ZNCCProcessor() {    
}

void ZNCCProcessor::setCameraMatrices(const float* H0_0, const float* K1inv_0, const float* e2_0) {
    setupCameraMatrices(H0_0, K1inv_0, e2_0);
}

void ZNCCProcessor::setCameraMatricesSecond(const float* H0_1, const float* K1inv_1, const float* e2_1) {    
    setupCameraMatricesSecond(H0_1, K1inv_1, e2_1);
}

void ZNCCProcessor::setNormalPermutations(const NormalPermutation* normalPermutations, int count) {    
    normalPermutationsHost.assign(normalPermutations, normalPermutations + count); // keep the host copy to reconstruct hypothesis normals after the kernel run
    setupNormalPermutations(normalPermutations, count);
}

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


        const float lambdaDepth = 0.05f;
        const float minRelativeDepthScale = 0.01f;
        const float maxNormalizedDeviation = 3.0f;
        const int minNeighborsForRegularization = 4;

        for (int f = 0; f < frontierCount; ++f)
        {
            float bestRegularizedScore = -std::numeric_limits<float>::infinity();
            float bestPhotoScore = -std::numeric_limits<float>::infinity();

            int bestDepthIdx = 0;
            int bestNormalIdx = 0;

            const int baseIdx = f * resultFloatsPerFrontier;

            std::vector<float> candidateLocalDepths;
            candidateLocalDepths.reserve((2 * localDepthWindowOffset + 1) * (2 * localDepthWindowOffset + 1) - 1);

            for (int offsetx = -localDepthWindowOffset;
                offsetx <= localDepthWindowOffset;
                ++offsetx)
            {
                for (int offsety = -localDepthWindowOffset;
                    offsety <= localDepthWindowOffset;
                    ++offsety)
                {
                    if (offsetx == 0 && offsety == 0)
                        continue;

                    const int neighborX = frontierPoints[f].x + offsetx;
                    const int neighborY = frontierPoints[f].y + offsety;

                    if (neighborX < 0 || neighborX >= depthMap.cols || neighborY < 0 || neighborY >= depthMap.rows)
                        continue;
                    
                    const float neighborDepth = depthMap.at<float>(neighborY, neighborX);

                    if (std::isfinite(neighborDepth) && neighborDepth > 0.f)
                        candidateLocalDepths.push_back(neighborDepth);
                }
            }

            const int localDepthCount = static_cast<int>(candidateLocalDepths.size());
            float localDepthMedian = frontierPoints[f].depth;
            float localDepthMAD = 0.f;
            float localDepthScale = 0.f;

            if (localDepthCount >= minNeighborsForRegularization)
            {
                std::sort(candidateLocalDepths.begin(), candidateLocalDepths.end());

                const size_t middle = candidateLocalDepths.size() / 2;

                if (candidateLocalDepths.size() % 2 == 0)                
                    localDepthMedian = 0.5f * (candidateLocalDepths[middle - 1] + candidateLocalDepths[middle]);                
                else                
                    localDepthMedian = candidateLocalDepths[middle];
                

                std::vector<float> deviations;
                deviations.reserve(candidateLocalDepths.size());

                for (const float depth : candidateLocalDepths)
                {
                    deviations.push_back(std::fabs(depth - localDepthMedian));
                }

                std::sort(deviations.begin(), deviations.end());

                const size_t deviationMiddle = deviations.size() / 2;

                if (deviations.size() % 2 == 0)                
                    localDepthMAD = 0.5f * (deviations[deviationMiddle - 1] + deviations[deviationMiddle]);                
                else                
                    localDepthMAD = deviations[deviationMiddle];                

                const float minDepthScale = minRelativeDepthScale * localDepthMedian;

                localDepthScale = std::max( 1.4826f * localDepthMAD, minDepthScale);
            }

            for (int d = 0; d < depthCount; ++d)
            {
                const int depthBase = baseIdx + d * normalNSteps;

                const float depthHypothesis = frontierPoints[f].depth + depthSteps[d];

                float depthPenalty = 0.f;

                if (localDepthCount >= minNeighborsForRegularization)
                {
                    const float normalizedDeviation = std::fabs(depthHypothesis - localDepthMedian) / localDepthScale;

                    depthPenalty = lambdaDepth * std::min(normalizedDeviation, maxNormalizedDeviation);
                }

                for (int n = 0; n < normalNSteps; ++n)
                {
                    const float score0 = resultHost0[depthBase + n];
                    const float score1 = resultHost1[depthBase + n];

                    if (!std::isfinite(score0) || !std::isfinite(score1)) continue;

                    const float photoScore = 0.5f * (score0 + score1);

                    const float regularizedScore = photoScore - depthPenalty;

					float normalCamDot = 0.f;

                    if (regularizedScore > bestRegularizedScore)
                    {
                        bestRegularizedScore = regularizedScore;
                        bestPhotoScore = photoScore;

                        bestDepthIdx = d;
                        bestNormalIdx = n;
                    }
                }
            }

            if (!std::isfinite(bestRegularizedScore)) continue;

            if (bestPhotoScore < znccThreshold) continue;

            ResultPoint rp;
            rp.x = frontierPoints[f].x;
            rp.y = frontierPoints[f].y;
            rp.depth = frontierPoints[f].depth + depthSteps[bestDepthIdx];
            rp.znccScore = bestPhotoScore;
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



