#include "../include/cuda_homography.h"
#include <stdio.h>
#include "../include/helper_math.h"


// calculate the total size of constant memory used sometime
__constant__ float H0[9];                         // K2 * R * K1^-1
__constant__ float K1inv[9];                      // K1^-1
__constant__ float e2[3];                         // K2 * t
__constant__ int windowRadius;
__constant__ NormalPermutation normalPermutations[16];
__constant__ size_t refPitch;
__constant__ size_t targetPitch;

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
bool projectSample(float3 referencePoint, float depth, float3 normal, float2& xy) {
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
    xy = make_float2(p.x / p.z, p.y / p.z);
    return true;
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

__device__
float interpolateTarget(const float* targetImg, int width, int height, float2 uv, size_t pitch) {
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
            rowValue += wx[i] * sampleClamp(targetImg, width, height, cx, cy, pitch);
        }

        value += wy[j] * rowValue;
    }

    return value;
}

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

    // Required: the launch must provide at least
    // depthNSteps * normalNSteps threads per block.
    if (depthIndex >= depthNSteps)
        return;

    FrontierPoint parent = frontier[frontierIndex];

    // Camera-axis Z depth of the center point for this depth hypothesis.
    //const float depth = parent.depth * depthSteps[depthIndex];
    const float depth = parent.depth + depthSteps[depthIndex];

    if (depth <= 0.0f)
        return;

    const int refX = parent.x;
    const int refY = parent.y;

    NormalPermutation perm = normalPermutations[normalIndex];

    float3 parentNormal = normalize(parent.normal);

    float3 parentTangent;
    float3 parentBitangent;
    float3 reference;

    if (fabsf(parentNormal.z) < 0.999f) {
        reference = make_float3(0.0f, 0.0f, 1.0f);
    }
    else {
        reference = make_float3(0.0f, 1.0f, 0.0f);
    }

    // Build an orthonormal tangent frame around the parent normal.
    parentTangent = normalize(cross(reference, parentNormal));
    parentBitangent = normalize(cross(parentNormal, parentTangent));

    float3 normVec = normalize(
        parentNormal
        + perm.tangent * parentTangent
        + perm.bitangent * parentBitangent
    );

    // ------------------------------------------------------------
    // Reconstruct the fixed 3D pivot point at the center of the
    // reference window.
    //
    // Assumption: depth is camera-axis Z depth in camera 1 space.
    // K1inv maps pixels to rays in camera 1 coordinates.
    // ------------------------------------------------------------

    float3 centerPixel = make_float3(
        static_cast<float>(refX),
        static_cast<float>(refY),
        1.0f
    );

    float3 centerRay = matmul3x3_point(K1inv, centerPixel);

    // A conventional pinhole camera produces centerRay.z ~= 1.
    // Divide by ray.z to support a general invertible K matrix.
    if (fabsf(centerRay.z) < 1e-8f)
        return;

    float rayScale = depth / centerRay.z;

    float3 pivotPoint = make_float3(
        centerRay.x * rayScale,
        centerRay.y * rayScale,
        centerRay.z * rayScale
    );

    const float planeOffset =
        normVec.x * pivotPoint.x
        + normVec.y * pivotPoint.y
        + normVec.z * pivotPoint.z;

    if (fabsf(planeOffset) < 1e-8f)
        return;

    const int windowSize = 1 + 2 * windowRadius;

    const int xStride = 2;

    const int yStride = windowSize * xStride;

    const int normalStride = windowSize * yStride;

    const int depthStride = normalNSteps * normalStride;

    const int frontierStride = depthNSteps * depthStride;
    
    //

    float sum_ref = 0.0f;
    float sum_target = 0.0f;
    float sum_sq_ref = 0.0f;
    float sum_sq_target = 0.0f;
    float sum_product = 0.0f;

    for (int winY = -windowRadius; winY <= windowRadius; ++winY)        
    {
		const int py = refY + winY;
        const float refPtYf = static_cast<float>(py);

        for (int winX = -windowRadius; winX <= windowRadius; ++winX)
        {
			const int px = refX + winX;
            const float refPtXf = static_cast<float>(px);

            float3 refPt = make_float3(refPtXf, refPtYf, 1.0f);

            float2 xy;

            bool valid = projectSample(refPt, planeOffset, normVec, xy);

            const int winXIndex = winX + windowRadius;

            const int winYIndex = winY + windowRadius;

            const int resultIdx =
                frontierIndex * frontierStride
                + depthIndex * depthStride
                + normalIndex * normalStride
                + winYIndex * yStride
                + winXIndex * xStride;

			float2 uv = valid ? make_float2(xy.x, xy.y) : make_float2(-1.f, -1.f);

            float* refRow = (float*)((char*)refImg + py * refPitch);
            float ref_val = refRow[px];
			
            float target_val = interpolateTarget(targetImg, targetWidth, targetHeight, uv, targetPitch);

            accumulateZNCC(sum_ref, sum_target, sum_sq_ref, sum_sq_target, sum_product, ref_val, target_val);

        }
    }

	float zncc = finishZNCC(sum_ref, sum_target, sum_sq_ref, sum_sq_target, sum_product, windowSize * windowSize);
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
    // safety: ensure we don't overrun device constant array
    const int maxPerms = static_cast<int>(sizeof(normalPermutations) / sizeof(normalPermutations[0]));
    if (count > maxPerms) {
        fprintf(stderr, "setupNormalPermutations: count (%d) exceeds device capacity (%d). Truncating to %d.\n", count, maxPerms, maxPerms);
        count = maxPerms;
    }
    // copy proper bytes based on the struct size
    cudaMemcpyToSymbol(normalPermutations, normalPermutations_host, count * sizeof(NormalPermutation));
    checkCudaError("setupNormalPermutations");
}

void setWindowRadius(int radius) {
    cudaMemcpyToSymbol(windowRadius, &radius, sizeof(int));
    checkCudaError("setWindowRadius");
}

// host helper to set image pitches in device constant memory
void setImagePitches(size_t refPitch_host, size_t targetPitch_host) {
    cudaMemcpyToSymbol(refPitch, &refPitch_host, sizeof(size_t));
    cudaMemcpyToSymbol(targetPitch, &targetPitch_host, sizeof(size_t));
    checkCudaError("setImagePitches");
}

void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", message, cudaGetErrorString(err));
    }
}