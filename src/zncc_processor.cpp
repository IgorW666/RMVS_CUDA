#include "../include/zncc_processor.h"
#include "../include/cuda_homography.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath> // for std::round, sqrtf
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <utility> // for std::pair


constexpr float znccThreshold = 0.75f; // threshold for ZNCC score to consider a hypothesis valid (for all score actually)
constexpr bool induceDepthDevPenalty = true; // whether to apply depth deviation penalty in ZNCC evaluation
constexpr float depthDevPenaltyStrength = 1.0f; // penalty factor for depth deviation (1 = no change) , too much and the clouds will be "stepped"

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

void ZNCCProcessor::setCameraMatricesSecond(const float* H1, const float* K1inv1, const float* e2_1) {
    // non-breaking addition: forwards to new device wrapper
    setupCameraMatricesSecond(H1, K1inv1, e2_1);
}

void ZNCCProcessor::setNormalPermutations(const NormalPermutation* normalPermutations, int count) {
    // keep a host-side copy so we can reconstruct hypothesis normals after the kernel run
    normalPermutationsHost.assign(normalPermutations, normalPermutations + count);

    // copy to device constant memory for the kernel
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

std::vector<ResultPoint> ZNCCProcessor::computeZNCC(
    const ImageData& refImage,
    const ImageData& targetImage,
    const std::vector<FrontierPoint>& frontierPoints,
    const std::vector<float>& depthSteps,
    int normalNSteps) {

    const int frontierCount = static_cast<int>(frontierPoints.size());
    if (frontierCount == 0 || depthSteps.empty() || normalNSteps <= 0) {
        return {};
    }

    // Prepare device-side buffers and ensure cleanup on error/exit
    FrontierPoint* frontierPtr = nullptr;
    float* depthStepsPtr = nullptr;
    float* resultDev = nullptr;

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

        setImagePitches(refImage.pitch, targetImage.pitch);

        // result buffer on device
        if (cudaMalloc((void**)&resultDev, resultFloatsTotal * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc resultDev failed");

        // launch kernel: one block per frontier point, threads = depthCount * normalNSteps
        launchEvaluateZNCC(
            refImage.gpuData,
            targetImage.gpuData,
            frontierPtr,
            depthStepsPtr,
            resultDev,
            depthCount,
            normalNSteps,
            refImage.width,
            refImage.height,
            targetImage.width,
            targetImage.height,
            frontierCount);

        checkCudaError("computeZNCC kernel launch");

        // copy results back to host
        std::vector<float> resultHost(resultFloatsTotal);
        if (cudaMemcpy(resultHost.data(), resultDev, resultFloatsTotal * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy resultDev DeviceToHost failed");

        // Repack results: for each frontier point pick the best (max) ZNCC across depth & normal hypotheses
        std::vector<ResultPoint> packedResults;
        packedResults.reserve(frontierCount);

        for (int f = 0; f < frontierCount; ++f) {
            float bestScore = -std::numeric_limits<float>::infinity();
            int bestDepthIdx = 0;
            int bestNormalIdx = 0;

            const int baseIdx = f * resultFloatsPerFrontier;
            for (int d = 0; d < depthCount; ++d) {
                const int depthBase = baseIdx + d * normalNSteps;
                for (int n = 0; n < normalNSteps; ++n) {
                    float score = resultHost[depthBase + n];
                    // skip NaNs
                    if (std::isnan(score)) continue;

                    const float depthDeviationPenalty = 1.f - depthDevPenaltyStrength * std::abs(depthSteps[d]);
					score *= depthDeviationPenalty;

                    if (score > bestScore) {
                        bestScore = score;
                        bestDepthIdx = d;
                        bestNormalIdx = n;
                    }
                }
            }

			// it shouldnt be here, it has to be calculated for all the hypotheses
            //if (induceDepthDevPenalty) { 
            //    const float depthDeviationPenalty = 1.f - depthDevPenaltyStrength * std::abs(depthSteps[bestDepthIdx]);
            //    bestScore = bestScore * depthDeviationPenalty;
            //}

            // If no valid score was found (all -inf/NaN), skip this frontier point
            if (!std::isfinite(bestScore) || bestScore < znccThreshold) {
                continue;
            }

            ResultPoint rp;
            rp.x = frontierPoints[f].x;
            rp.y = frontierPoints[f].y;
            rp.depth = frontierPoints[f].depth + depthSteps[bestDepthIdx];
            rp.znccScore = bestScore;

            // Reconstruct hypothesis normal on host using the same method as the device kernel
            // If host-side normal permutations were not provided, fall back to parent normal.
            if (bestNormalIdx >= 0 && bestNormalIdx < static_cast<int>(normalPermutationsHost.size())) {
                NormalPermutation perm = normalPermutationsHost[bestNormalIdx];

                float3 parentNormal = host_normalize(frontierPoints[f].normal);

                float3 reference;
                if (std::fabs(parentNormal.z) < 0.999f) {
                    reference = make_float3(0.0f, 0.0f, 1.0f);
                }
                else {
                    reference = make_float3(0.0f, 1.0f, 0.0f);
                }

                float3 parentTangent = host_normalize(host_cross(reference, parentNormal));
                float3 parentBitangent = host_normalize(host_cross(parentNormal, parentTangent));

                float3 hypothesis = make_float3(
                    parentNormal.x + perm.tangent * parentTangent.x + perm.bitangent * parentBitangent.x,
                    parentNormal.y + perm.tangent * parentTangent.y + perm.bitangent * parentBitangent.y,
                    parentNormal.z + perm.tangent * parentTangent.z + perm.bitangent * parentBitangent.z
                );

                rp.normal = host_normalize(hypothesis);
            }
            else {
                // fallback
                rp.normal = host_normalize(frontierPoints[f].normal);
            }

            packedResults.push_back(rp);
        }

        // cleanup
        cudaFree(frontierPtr);
        cudaFree(depthStepsPtr);
        cudaFree(resultDev);

        //std::cout << "computeZNCC: Frontier points processed: " << frontierCount << ", valid results: " << packedResults.size() << std::endl;

        return packedResults;
    }
    catch (...) {
        // Ensure resources are freed on any exception
        if (frontierPtr) cudaFree(frontierPtr);
        if (depthStepsPtr) cudaFree(depthStepsPtr);
        if (resultDev) cudaFree(resultDev);
        throw;
    }
}


std::vector<ResultPoint> ZNCCProcessor::computeZNCCDualTargets(
    const ImageData& refImage,
    const ImageData& targetImage0,
    const ImageData& targetImage1,
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

        // Repack results requiring mutual agreement between target0 and target1.
        // Agreement criteria:
        //  - both targets select finite scores >= znccThreshold
        //  - both targets choose the same depth index AND the same normal index
        std::vector<ResultPoint> consensusResults;
        consensusResults.reserve(frontierCount);

        auto reconstruct_normal = [&](int fIdx, int normalIdx) -> float3 {
            if (normalIdx >= 0 && normalIdx < static_cast<int>(normalPermutationsHost.size())) {
                NormalPermutation perm = normalPermutationsHost[normalIdx];

                float3 parentNormal = host_normalize(frontierPoints[fIdx].normal);

                float3 reference;
                if (std::fabs(parentNormal.z) < 0.999f) {
                    reference = make_float3(0.0f, 0.0f, 1.0f);
                }
                else {
                    reference = make_float3(0.0f, 1.0f, 0.0f);
                }

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

            for (int d = 0; d < depthCount; ++d) {
                const int depthBase = baseIdx + d * normalNSteps;

                for (int n = 0; n < normalNSteps; ++n) {
                    const float s0 = resultHost0[depthBase + n];
                    const float s1 = resultHost1[depthBase + n];

                    // A joint hypothesis requires valid support from both targets.
                    if (!std::isfinite(s0) || !std::isfinite(s1))
                        continue;

                    float jointScore = 0.5f * (s0 + s1);

                    // Apply depth deviation penalty to the hypothesis itself.
                    if (induceDepthDevPenalty) {
                        const float penalty =
                            1.f - depthDevPenaltyStrength *
                            std::abs(depthSteps[d]);

                        jointScore *= penalty;
                    }

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


        /*for (int f = 0; f < frontierCount; ++f) {
            // find best hypothesis for target 0
            float bestScore0 = -std::numeric_limits<float>::infinity();
            int bestDepthIdx0 = 0;
            int bestNormalIdx0 = 0;

            // find best hypothesis for target 1
            float bestScore1 = -std::numeric_limits<float>::infinity();
            int bestDepthIdx1 = 0;
            int bestNormalIdx1 = 0;

            const int baseIdx = f * resultFloatsPerFrontier;
            for (int d = 0; d < depthCount; ++d) {
                const int depthBase = baseIdx + d * normalNSteps;
                for (int n = 0; n < normalNSteps; ++n) {
                    float s0 = resultHost0[depthBase + n];
                    if (!std::isnan(s0) && s0 > bestScore0) {
                        bestScore0 = s0;
                        bestDepthIdx0 = d;
                        bestNormalIdx0 = n;
                    }

                    float s1 = resultHost1[depthBase + n];
                    if (!std::isnan(s1) && s1 > bestScore1) {
                        bestScore1 = s1;
                        bestDepthIdx1 = d;
                        bestNormalIdx1 = n;
                    }
                }
            }

            // apply optional depth deviation penalty independently
            if (induceDepthDevPenalty) {
                const float depthDeviationPenalty0 = 1.f - depthDevPenaltyStrength * std::abs(depthSteps[bestDepthIdx0]);
                bestScore0 = bestScore0 * depthDeviationPenalty0;

                const float depthDeviationPenalty1 = 1.f - depthDevPenaltyStrength * std::abs(depthSteps[bestDepthIdx1]);
                bestScore1 = bestScore1 * depthDeviationPenalty1;
            }

            // require both scores to be finite and above threshold
            if (!std::isfinite(bestScore0) || !std::isfinite(bestScore1)) continue;
            if (bestScore0 < znccThreshold || bestScore1 < znccThreshold) continue;

            // require exact agreement on depth index and normal index (mutual consistency)
            if (bestDepthIdx0 == bestDepthIdx1 && bestNormalIdx0 == bestNormalIdx1) {
                ResultPoint rp;
                rp.x = frontierPoints[f].x;
                rp.y = frontierPoints[f].y;
                rp.depth = frontierPoints[f].depth + depthSteps[bestDepthIdx0];
                rp.znccScore = 0.5f * (bestScore0 + bestScore1);
                rp.normal = reconstruct_normal(f, bestNormalIdx0);

                consensusResults.push_back(rp);
            }
            // else: disagreement => ignore (no consensus)
        }*/

        // cleanup
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