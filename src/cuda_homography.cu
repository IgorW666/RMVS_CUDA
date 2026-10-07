#include "../include/cuda_homography.h"
#include <stdio.h>

__constant__ float H0[9];     // K2 * R * K1^-1
__constant__ float K1inv[9];  // K1^-1
__constant__ float e2[3];     // K2 * t
__constant__ int windowRadius;

__device__
inline float3 matmul3x3_point(const float* mat, float3 p) {
    return make_float3(
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z,
        mat[3] * p.x + mat[4] * p.y + mat[5] * p.z,
        mat[6] * p.x + mat[7] * p.y + mat[8] * p.z
    );
}

// project a reference pixel to the target image using the plane-induced homography:
// p' = H0 * p + (K2 * t) * (n^T * K1^-1 * p) / d
// returns false if the point is behind the target camera
__device__
bool projectSample(float3 referencePoint, float depth, float3 normal, float2& uv) {
    float3 q = matmul3x3_point(K1inv, referencePoint);
    float s = (normal.x * q.x + normal.y * q.y + normal.z * q.z) / depth;
    float3 h = matmul3x3_point(H0, referencePoint);

    float3 p = make_float3(h.x + e2[0] * s, h.y + e2[1] * s, h.z + e2[2] * s);
    if (p.z <= 1e-6f) return false;

    uv = make_float2(p.x / p.z, p.y / p.z);
    return true;
}

__global__
void evaluateZNCC(
    const float* refImg,
    const float* targetImg,
    const FrontierPoint* frontier, 
    float* result,
    float3 parentNorm,    
    int depthSteps,
    int normalSteps,
    int refWidth,
    int refHeight,
    int targetWidth,
    int targetHeight)
{

    const int frontierIndex = blockIdx.x;

    const int hypothesis = threadIdx.x;

    const int depthIndex = hypothesis / normalSteps;

    const int normalIndex = hypothesis % normalSteps;    

    FrontierPoint parent = frontier[frontierIndex];

    const float depth = parent.depth; // + depthIndex * 0.1; // need a depth step ?

    const int refX = parent.x;
    const int refY = parent.y;
    
    float3 normVec = make_float3(0.f, 0.f, 1.f); // Dummy normal for now.

    const int windowSize = 1 + 2 * windowRadius;

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

            const int resultIdx = (((depthIndex * normalSteps + normalIndex) * windowSize + winYIndex) * windowSize + winXIndex) * 2;

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

void launchEvaluateZNCC(float* refImage, float* targetImage, float* depthHypotheses, float* result,
    int refX, int refY, int depthSteps, int normalSteps,
    int refWidth, int refHeight, int targetWidth, int targetHeight) {
    dim3 blockDim(blockX, blockY);
    dim3 gridDim(gridX, gridY);
    evaluateZNCC <<<gridDim, blockDim>>> (
        refImg,
        targetImg,
        depthHypotheses,
        result,
        parentNorm,
        refX,
        refY,
        depthSteps,
        normalSteps,
        refWidth,
        refHeight,
        targetWidth,
        targetHeight);
}










void setupCameraMatrices(const float* h0_host, const float* k1inv_host, const float* e2_host) {
    cudaMemcpyToSymbol(H0, h0_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(K1inv, k1inv_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(e2, e2_host, 3 * sizeof(float));
    checkCudaError("setupCameraMatrices");
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