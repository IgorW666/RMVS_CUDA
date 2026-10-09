#include "../include/cuda_homography.h"
#include <stdio.h>

__constant__ float H0[9];                         // K2 * R * K1^-1
__constant__ float K1inv[9];                      // K1^-1
__constant__ float e2[3];                         // K2 * t
__constant__ int windowRadius;
__constant__ float normalPermutations[16][9];     // 16 permutations * 3x3 rotation matrix (this should NOT be constant later) << or maybe it should. Depends on how the kernel will be used

__device__
inline float3 matmul3x3_point(const float* mat, float3 p) {
    return make_float3(
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z,
        mat[3] * p.x + mat[4] * p.y + mat[5] * p.z,
        mat[6] * p.x + mat[7] * p.y + mat[8] * p.z
    );
}


// standard homography transformed for precalculated H0, K1inv, e
__device__
bool projectSample(float3 referencePoint, float depth, float3 normal, float2& uv) {
    float3 q = matmul3x3_point(K1inv, referencePoint);

    float s = (normal.x * q.x + normal.y * q.y + normal.z * q.z) / depth;

    float3 h = matmul3x3_point(H0, referencePoint);

    float3 p = make_float3(
        h.x + e2[0] * s,
        h.y + e2[1] * s,
        h.z + e2[2] * s
    );

    if (p.z <= 1e-6f)
        return false;

    uv = make_float2(p.x / p.z, p.y / p.z);

    return true;
}


__global__
void evaluateZNCC(
    const float* refImg,
    const float* targetImg,
    const FrontierPoint* frontier,   // array with frontier points with depth and normal inherited from parent but new {x,y} <int>
    const float* depthSteps,         // array with depth values for that frontier point eg {-0.98, -0.99, 1.00, 1.01, 1.02 } but 8 or 16 values
    float* result,
    int depthNSteps,
    int normalNSteps,
    int refWidth,
    int refHeight,
    int targetWidth,
    int targetHeight)
{
    const int frontierIndex = blockIdx.x;

    const int hypothesis = threadIdx.x;

    const int depthIndex = hypothesis / normalNSteps;

    const int normalIndex = hypothesis % normalNSteps;

    FrontierPoint parent = frontier[frontierIndex];

    const float depth = parent.depth * depthSteps[depthIndex];

    const int refX = parent.x;
    const int refY = parent.y;

    const float* normPermutation = normalPermutations[normalIndex];

    float3 normVec = matmul3x3_point(normPermutation, parent.normal);

    const int windowSize = 1 + 2 * windowRadius;

    // Result layout:
    //
    // frontier
    //     -> depth
    //         -> normal
    //             -> window Y
    //                 -> window X
    //                     -> {u, v}
    //
    // Each frontier point therefore gets its own complete result region.

    const int xStride = 2;

    const int yStride = windowSize * xStride;

    const int normalStride = windowSize * yStride;

    const int depthStride = normalNSteps * normalStride;

    const int frontierStride = depthNSteps * depthStride;

    for (int winY = -windowRadius; winY <= windowRadius; ++winY)
    {
        const float refPtYf = static_cast<float>(refY + winY);

        for (int winX = -windowRadius; winX <= windowRadius; ++winX)
        {
            const float refPtXf = static_cast<float>(refX + winX);

            float3 refPt = make_float3(refPtXf, refPtYf, 1.0f);

            float2 uv;

            bool valid = projectSample(refPt, depth, normVec, uv);

            const int winXIndex = winX + windowRadius;

            const int winYIndex = winY + windowRadius;

            const int resultIdx =
                frontierIndex * frontierStride
                + depthIndex * depthStride
                + normalIndex * normalStride
                + winYIndex * yStride
                + winXIndex * xStride;

            if (valid)
            {
                result[resultIdx + 0] = uv.x;
                result[resultIdx + 1] = uv.y;
            }
            else
            {
                result[resultIdx + 0] = -1.f;
                result[resultIdx + 1] = -1.f;
            }
        }
    }
}

void launchEvaluateZNCC(const float* refImg,
    const float* targetImg,
    const FrontierPoint* frontier,   // array with frontier points with depth and normal inherited from parent but new {x,y} <int>
    const float* depthSteps,         // array with depth values for that frontier point eg {-0.98, -0.99, 1.00, 1.01, 1.02 } but 8 or 16 values
    float* result,
    int depthNSteps,
    int normalNSteps,
    int refWidth,
    int refHeight,
    int targetWidth,
    int targetHeight,
    int frontierCount) { // <-- added frontierCount parameter
    
    // Use frontierCount as number of blocks (one block per frontier point)
    evaluateZNCC <<<frontierCount, depthNSteps * normalNSteps>>> (
        refImg,
        targetImg,
        frontier,
        depthSteps,
        result,
        depthNSteps,
        normalNSteps,
        refWidth,
        refHeight,
        targetWidth,
        targetHeight);
}

void setupCameraMatrices(const float* h0_host, const float* k1inv_host, const float* e2_host) {
    cudaMemcpyToSymbol(H0, h0_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(K1inv, k1inv_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(e2, e2_host, 3 * sizeof(float));
    checkCudaError("setupCameraMatrices"); // handy!
}

void setupNormalPermutations(const NormalPermutation* normalPermutations_host, int count) {
    cudaMemcpyToSymbol(normalPermutations, normalPermutations_host, count * 9 * sizeof(float));
    checkCudaError("setupNormalPermutations");
}

void setWindowRadius(int radius) {
    cudaMemcpyToSymbol(windowRadius, &radius, sizeof(int));
    checkCudaError("setWindowRadius");
}

void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", message, cudaGetErrorString(err));
    }
}