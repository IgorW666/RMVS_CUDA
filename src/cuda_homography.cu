#include <stdio.h>
#include "../include/helper_math.h"
#include "../include/cuda_homography.h"

// calculate the total size of constant memory used sometime
__constant__ float H0_0[9];                         // K2 * R * K1^-1  << this should be H0_0
__constant__ float K1inv_0[9];                      // K1^-1
__constant__ float e2_0[3];                         // K2 * t
__constant__ int windowRadius;
__constant__ NormalPermutation normalPermutations[16];
__constant__ size_t refPitch;
__constant__ size_t targetPitch;

// new constants for second target camera
__constant__ float H0_1[9];                         // second target: K2' * R' * K1^-1 << this should be H0_1
__constant__ float K1inv_1[9];                     // (optional) if a different K1inv needed; typically same as K1inv
__constant__ float e2_1[3];                       // K2' * t'
__constant__ size_t target2Pitch;


// device helper functions remain unchanged
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
    float3 q = matmul3x3_point(K1inv_0, referencePoint);
    float s = (normal.x * q.x + normal.y * q.y + normal.z * q.z) / depth;
    float3 h = matmul3x3_point(H0_0, referencePoint);
    float3 p = make_float3(
        h.x + e2_0[0] * s,
        h.y + e2_0[1] * s,
        h.z + e2_0[2] * s
    );
    if (p.z <= 1e-6f)
        return false;
    xy = make_float2(p.x / p.z, p.y / p.z);
    return true;
}

// Generic projectSample that accepts explicit camera constants (for dual-target kernel)
__device__
bool projectSampleGeneric(const float* H, const float* Kinv_local, const float* e_local,
    float3 referencePoint, float depth, float3 normal, float2& xy)
{
    float3 q = matmul3x3_point(Kinv_local, referencePoint);
    float s = (normal.x * q.x + normal.y * q.y + normal.z * q.z) / depth;
    float3 h = matmul3x3_point(H, referencePoint);
    float3 p = make_float3(
        h.x + e_local[0] * s,
        h.y + e_local[1] * s,
        h.z + e_local[2] * s
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

// i should really start paying Wikipedia some tribute
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



/// Dual-target kernel: computes ZNCC for two target images per hypothesis
__global__
void evaluateZNCC_DualTargets(
    const float* refImg,
    const float* targetImg0,
    const float* targetImg1,
    const FrontierPoint* frontier,
    const float* depthSteps,
    float* result0,
    float* result1,
    int depthNSteps,
    int normalNSteps,
    int refWidth,
    int refHeight,
    int target0Width,
    int target0Height,
    int target1Width,
    int target1Height)
{
    const int frontierIndex = blockIdx.x;
    const int hypothesis = threadIdx.x;
    const int depthIndex = hypothesis / normalNSteps;
    const int normalIndex = hypothesis % normalNSteps;

    if (depthIndex >= depthNSteps) return; // investigate kernel behavior on return

    FrontierPoint parent = frontier[frontierIndex];

    const float depth = parent.depth + depthSteps[depthIndex];

	if (depth <= 0.0f) return;  // this should not happen. check and remove

    const int refX = parent.x;
    const int refY = parent.y;

    NormalPermutation perm = normalPermutations[normalIndex];

    float3 parentNormal = normalize(parent.normal);

    float3 parentTangent;
    float3 parentBitangent;
    float3 reference;

	// this was supposed to keep the math happy. investigate and remove/replace if possible
	// maybe setting the initial normals to slightly off-axis would be better than this?
    // result normals on dense clouds sometimes look like "wet hair" instead of spikes
    // commenting this out breaks the reconstruction 
    if (fabsf(parentNormal.z) < 0.999f) {
        reference = make_float3(0.0f, 0.0f, 1.0f);
    }
    else {
        reference = make_float3(0.0f, 1.0f, 0.0f);
    }

    parentTangent = normalize(cross(reference, parentNormal));
    parentBitangent = normalize(cross(parentNormal, parentTangent));

    float3 normVec = normalize(
        parentNormal
        + perm.tangent * parentTangent
        + perm.bitangent * parentBitangent
    );

    float3 centerPixel = make_float3((float)refX, (float)refY, 1.0f);
    float3 centerRay = matmul3x3_point(K1inv_0, centerPixel);
    if (fabsf(centerRay.z) < 1e-8f) return;
    float rayScale = depth / centerRay.z;
    float3 pivotPoint = make_float3(centerRay.x * rayScale, centerRay.y * rayScale, centerRay.z * rayScale);

    const float planeOffset =
        normVec.x * pivotPoint.x
        + normVec.y * pivotPoint.y
        + normVec.z * pivotPoint.z;

    if (fabsf(planeOffset) < 1e-8f)
        return;

    const int windowSize = 1 + 2 * windowRadius; // move to __constant__ ?

    float sum_ref = 0.0f;
    float sum_sq_ref = 0.0f;

    float sum_target0 = 0.0f;
    float sum_sq_target0 = 0.0f;
    float sum_product0 = 0.0f;

    float sum_target1 = 0.0f;
    float sum_sq_target1 = 0.0f;
    float sum_product1 = 0.0f;

    for (int winY = -windowRadius; winY <= windowRadius; ++winY)
    {
        const int py = refY + winY;
        const float refPtYf = static_cast<float>(py);

        for (int winX = -windowRadius; winX <= windowRadius; ++winX)
        {
            const int px = refX + winX;
            const float refPtXf = static_cast<float>(px);

            float3 refPt = make_float3(refPtXf, refPtYf, 1.0f);

            float2 uv0;
            float2 uv1;

            bool valid0 = projectSampleGeneric(H0_0, K1inv_0, e2_0, refPt, planeOffset, normVec, uv0);
            bool valid1 = projectSampleGeneric(H0_1, K1inv_1, e2_1, refPt, planeOffset, normVec, uv1);

            float2 sampleUV0 = valid0 ? uv0 : make_float2(-1.f, -1.f);
            float2 sampleUV1 = valid1 ? uv1 : make_float2(-1.f, -1.f);

            float* refRow = (float*)((char*)refImg + py * refPitch);
            float ref_val = refRow[px];

            float target_val0 = interpolateTarget(targetImg0, target0Width, target0Height, sampleUV0, targetPitch);
            float target_val1 = interpolateTarget(targetImg1, target1Width, target1Height, sampleUV1, target2Pitch);

            // update reference accumulators
            sum_ref += ref_val;
            sum_sq_ref += ref_val * ref_val;

            // update target 0 accumulators
            sum_target0 += target_val0;
            sum_sq_target0 += target_val0 * target_val0;
            sum_product0 += ref_val * target_val0;

            // update target 1 accumulators
            sum_target1 += target_val1;
            sum_sq_target1 += target_val1 * target_val1;
            sum_product1 += ref_val * target_val1;
        }
    }

    const int count = windowSize * windowSize;

    // finish ZNCC for both targets; finishZNCC expects full ref sums, so we pass them accordingly
    float zncc0;
    {
        float mean_ref = sum_ref / (float)count;
        float mean_target = sum_target0 / (float)count;
        float var_ref = sum_sq_ref / (float)count - mean_ref * mean_ref;
        float var_target = sum_sq_target0 / (float)count - mean_target * mean_target;
        float cov = sum_product0 / (float)count - mean_ref * mean_target;
        var_ref = max(1e-6f, var_ref);
        var_target = max(1e-6f, var_target);
        zncc0 = cov / sqrtf(var_ref * var_target);
    }

    float zncc1;
    {
        float mean_ref = sum_ref / (float)count;
        float mean_target = sum_target1 / (float)count;
        float var_ref = sum_sq_ref / (float)count - mean_ref * mean_ref;
        float var_target = sum_sq_target1 / (float)count - mean_target * mean_target;
        float cov = sum_product1 / (float)count - mean_ref * mean_target;
        var_ref = max(1e-6f, var_ref);
        var_target = max(1e-6f, var_target);
        zncc1 = cov / sqrtf(var_ref * var_target);
    }

    int resultIdx =
        frontierIndex * (depthNSteps * normalNSteps)
        + depthIndex * normalNSteps
        + normalIndex;

    result0[resultIdx] = zncc0;
    result1[resultIdx] = zncc1;
}

void launchEvaluateZNCC_DualTargets(const float* refImg,
    const float* targetImg0,
    const float* targetImg1,
    const FrontierPoint* frontier,
    const float* depthSteps,
    float* result0,
    float* result1,
    int depthNSteps,
    int normalNSteps,
    int refWidth,
    int refHeight,
    int target0Width,
    int target0Height,
    int target1Width,
    int target1Height,
    int frontierCount)
{
    evaluateZNCC_DualTargets << <frontierCount, depthNSteps* normalNSteps >> > (
        refImg,
        targetImg0,
        targetImg1,
        frontier,
        depthSteps,
        result0,
        result1,
        depthNSteps,
        normalNSteps,
        refWidth,
        refHeight,
        target0Width,
        target0Height,
        target1Width,
        target1Height);
}

void setupCameraMatrices(const float* h0_0_host, const float* k1inv_0_host, const float* e2_0_host) {
    cudaMemcpyToSymbol(H0_0, h0_0_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(K1inv_0, k1inv_0_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(e2_0, e2_0_host, 3 * sizeof(float));
    checkCudaError("setupCameraMatrices"); // handy!
}

// copy second-camera matrices into device constant memory
void setupCameraMatricesSecond(const float* h1_0_host, const float* k1inv_0_host, const float* e2_0_host) {
    cudaMemcpyToSymbol(H0_1, h1_0_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(K1inv_1, k1inv_0_host, 9 * sizeof(float));
    cudaMemcpyToSymbol(e2_1, e2_0_host, 3 * sizeof(float));
    checkCudaError("setupCameraMatricesSecond");
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

// host helper to set 3 image pitches (ref, target0, target1)
void setImagePitchesDual(size_t refPitch_host, size_t targetPitch_host, size_t target2Pitch_host) {
    cudaMemcpyToSymbol(refPitch, &refPitch_host, sizeof(size_t));
    cudaMemcpyToSymbol(targetPitch, &targetPitch_host, sizeof(size_t));
    cudaMemcpyToSymbol(target2Pitch, &target2Pitch_host, sizeof(size_t));
    checkCudaError("setImagePitchesDual");
}

void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", message, cudaGetErrorString(err));
    }
}

// old single cam kernal
/*__global__
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

    // still better then transforming rotation matrices
    parentTangent = normalize(cross(reference, parentNormal));
    parentBitangent = normalize(cross(parentNormal, parentTangent));

    float3 normVec = normalize(
        parentNormal
        + perm.tangent * parentTangent
        + perm.bitangent * parentBitangent
    );

	// pivot calculation: center pixel > center ray > pivot point
    float3 centerPixel = make_float3(
        static_cast<float>(refX),
        static_cast<float>(refY),
        1.0f
    );

    // rays from RMVS
    float3 centerRay = matmul3x3_point(K1inv, centerPixel);
    if (fabsf(centerRay.z) < 1e-8f)
        return;

    float rayScale = depth / centerRay.z;

    float3 pivotPoint = make_float3(centerRay.x * rayScale, centerRay.y * rayScale, centerRay.z * rayScale);

    const float planeOffset = normVec.x * pivotPoint.x + normVec.y * pivotPoint.y + normVec.z * pivotPoint.z;

    if (fabsf(planeOffset) < 1e-8f) return;

    const int windowSize = 1 + 2 * windowRadius;
    const int xStride = 2;
    const int yStride = windowSize * xStride;
    const int normalStride = windowSize * yStride;
    const int depthStride = normalNSteps * normalStride;
    const int frontierStride = depthNSteps * depthStride;

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
            float2 uv = valid ? make_float2(xy.x, xy.y) : make_float2(-1.f, -1.f);
            float* refRow = (float*)((char*)refImg + py * refPitch);
            float ref_val = refRow[px];
            float target_val = interpolateTarget(targetImg, targetWidth, targetHeight, uv, targetPitch);
            accumulateZNCC(sum_ref, sum_target, sum_sq_ref, sum_sq_target, sum_product, ref_val, target_val);
        }
    }
    float zncc = finishZNCC(sum_ref, sum_target, sum_sq_ref, sum_sq_target, sum_product, windowSize * windowSize);

    int resultIdx =
        frontierIndex * (depthNSteps * normalNSteps)
        + depthIndex * normalNSteps
        + normalIndex;

    result[resultIdx] = zncc;
}
*/