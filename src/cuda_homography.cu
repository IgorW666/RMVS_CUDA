#include "../include/cuda_homography.h"
#include <stdio.h>

// ============================================================================
// Constant Memory Symbols
// ============================================================================
__constant__ float H0[9];     // K2 * R * K1^-1
__constant__ float K1inv[9];  // K1^-1
__constant__ float e2[3];     // K2 * t
__constant__ int windowRadius;

// ============================================================================
// Device Helper Functions (Device-only)
// ============================================================================

// Helper function to multiply 3x3 matrix by 3D point (row-major storage)
__device__
inline float3 matmul3x3_point(const float* mat, float3 p) {
    return make_float3(
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z,
        mat[3] * p.x + mat[4] * p.y + mat[5] * p.z,
        mat[6] * p.x + mat[7] * p.y + mat[8] * p.z
    );
}

// Catmull-Rom bicubic weights
__device__
inline void getBicubicWeights(float t, float* w) {
    float t2 = t * t;
    float t3 = t2 * t;

    w[0] = 0.5f * (-t3 + 2.0f * t2 - t);
    w[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
    w[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
    w[3] = 0.5f * (t3 - t2);
}

// Safely sample global memory with clamping at borders
__device__
inline float sampleClamp(const float* image, int width, int height, int x, int y, size_t pitch) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    const float* row = (const float*)((const char*)image + y * pitch);
    return row[x];
}

// Project a reference pixel to the target image using the plane-induced homography:
// p' = H0 * p + (K2 * t) * (n^T * K1^-1 * p) / d
// Returns false if the point is behind the target camera.
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

// Bicubic interpolation in target image (pixel centers at integer coordinates)
__device__
float interpolateTarget(float* targetImage, int width, int height, float2 uv, size_t pitch) {
    int ix = (int)floorf(uv.x);
    int iy = (int)floorf(uv.y);

    float fx = uv.x - (float)ix;
    float fy = uv.y - (float)iy;

    float wx[4], wy[4];
    getBicubicWeights(fx, wx);
    getBicubicWeights(fy, wy);

    float value = 0.0f;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float rowValue = 0.0f;
        int cy = iy + j - 1;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int cx = ix + i - 1;
            rowValue += wx[i] * sampleClamp(targetImage, width, height, cx, cy, pitch);
        }

        value += wy[j] * rowValue;
    }

    return value;
}

// Accumulate sums, squares and cross-product
__device__
void accumulateZNCC(float& sum_ref, float& sum_target,
    float& sum_sq_ref, float& sum_sq_target,
    float& sum_product, float ref_val, float target_val) {
    sum_ref += ref_val;
    sum_target += target_val;
    sum_sq_ref += ref_val * ref_val;
    sum_sq_target += target_val * target_val;
    sum_product += ref_val * target_val;
}

// Zero-mean normalized cross-correlation
__device__
float finishZNCC(float sum_ref, float sum_target, float sum_sq_ref,
    float sum_sq_target, float sum_product, int count) {
    if (count == 0) return 0.0f;

    float mean_ref = sum_ref / (float)count;
    float mean_target = sum_target / (float)count;

    float var_ref = sum_sq_ref / (float)count - mean_ref * mean_ref;
    float var_target = sum_sq_target / (float)count - mean_target * mean_target;
    float cov = sum_product / (float)count - mean_ref * mean_target;

    var_ref = max(1e-6f, var_ref);
    var_target = max(1e-6f, var_target);

    float denom = sqrtf(var_ref * var_target);
    return cov / denom;
}

// ============================================================================
// CUDA Kernel
// ============================================================================

__global__
void evaluateZNCC(float* refImage, float* targetImage, float* depthMap, float* normalMap,
    int refWidth, int refHeight, int targetWidth, int targetHeight, float* znccMat,
    size_t refPitch, size_t targetPitch, size_t depthPitch, size_t normalPitch, size_t outPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= refWidth || y >= refHeight) return;

    if (x == refWidth / 2 && y == refHeight / 2) {
        float* dRow = (float*)((char*)depthMap + y * depthPitch);
        float* nRow = (float*)((char*)normalMap + y * normalPitch);
        float3 rp = make_float3((float)x, (float)y, 1.0f);
        float3 q = matmul3x3_point(K1inv, rp);
        float3 h = matmul3x3_point(H0, rp);
        float2 uvDbg = make_float2(0.0f, 0.0f);
        bool ok = projectSample(rp, dRow[x], make_float3(nRow[x * 3], nRow[x * 3 + 1], nRow[x * 3 + 2]), uvDbg);

        printf("depth=%f n=(%f,%f,%f)\n", dRow[x], nRow[x * 3], nRow[x * 3 + 1], nRow[x * 3 + 2]);
        printf("H0*p=(%f,%f,%f) e2=(%f,%f,%f) K1inv*p=(%f,%f,%f)\n",
            h.x, h.y, h.z, e2[0], e2[1], e2[2], q.x, q.y, q.z);
        printf("ok=%d uv=(%f,%f) target=%dx%d\n", (int)ok, uvDbg.x, uvDbg.y, targetWidth, targetHeight);
    }

    float* depthRow = (float*)((char*)depthMap + y * depthPitch);
    float depth = depthRow[x];

    float* normalRow = (float*)((char*)normalMap + y * normalPitch);
    float3 normal = make_float3(normalRow[x * 3], normalRow[x * 3 + 1], normalRow[x * 3 + 2]);

    float sum_ref = 0.0f;
    float sum_target = 0.0f;
    float sum_sq_ref = 0.0f;
    float sum_sq_target = 0.0f;
    float sum_product = 0.0f;
    int count = 0;

    for (int oy = -windowRadius; oy <= windowRadius; ++oy) {
        for (int ox = -windowRadius; ox <= windowRadius; ++ox) {
            int py = y + oy;
            int px = x + ox;

            if (px < 0 || px >= refWidth || py < 0 || py >= refHeight) continue;

            float3 refPoint = make_float3((float)px, (float)py, 1.0f);
            float2 uv;
            if (!projectSample(refPoint, depth, normal, uv)) continue;
            if (uv.x < 0.0f || uv.x >(float)(targetWidth - 1) ||
                uv.y < 0.0f || uv.y >(float)(targetHeight - 1)) continue;

            float* refRow = (float*)((char*)refImage + py * refPitch);
            float ref_val = refRow[px];
            float target_val = interpolateTarget(targetImage, targetWidth, targetHeight, uv, targetPitch);

            accumulateZNCC(sum_ref, sum_target, sum_sq_ref, sum_sq_target, sum_product, ref_val, target_val);
            ++count;
        }
    }

    float* outRow = (float*)((char*)znccMat + y * outPitch);
    outRow[x] = finishZNCC(sum_ref, sum_target, sum_sq_ref, sum_sq_target, sum_product, count);
}

// ============================================================================
// Kernel Launch Wrapper (callable from .cpp files)
// ============================================================================

void launchEvaluateZNCC(float* refImage, float* targetImage, float* depthMap, float* normalMap,
    int refWidth, int refHeight, int targetWidth, int targetHeight, float* znccMat,
    size_t refPitch, size_t targetPitch, size_t depthPitch, size_t normalPitch,
    unsigned int blockX, unsigned int blockY, unsigned int gridX, unsigned int gridY,
    size_t outPitch) {
    dim3 blockDim(blockX, blockY);
    dim3 gridDim(gridX, gridY);
    evaluateZNCC << <gridDim, blockDim >> > (refImage, targetImage, depthMap, normalMap,
        refWidth, refHeight, targetWidth, targetHeight, znccMat,
        refPitch, targetPitch, depthPitch, normalPitch, outPitch);
}

// ============================================================================
// Host-side Setup Functions
// ============================================================================

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