#pragma once

#include <cuda_runtime.h>

// ============================================================================
// CUDA Kernel Launch Interface
// ============================================================================

/**
 * Evaluates ZNCC (Zero-mean Normalized Cross-Correlation) for depth map
 * using plane-induced homography and bicubic interpolation.
 *
 * @param refWidth/refHeight        Reference image size (also depth/normal/output size)
 * @param targetWidth/targetHeight  Target image size
 * @param *Pitch                    Row pitch in bytes of the respective buffer
 */
void launchEvaluateZNCC(float* refImage, float* targetImage, float* depthMap, float* normalMap,
    int refWidth, int refHeight, int targetWidth, int targetHeight, float* znccMat,
    size_t refPitch, size_t targetPitch, size_t depthPitch, size_t normalPitch,
    unsigned int blockX, unsigned int blockY, unsigned int gridX, unsigned int gridY,
    size_t outPitch);

/**
 * Evaluates ZNCC kernel (device code).
 */
__global__
void evaluateZNCC(float* refImage, float* targetImage, float* depthMap, float* normalMap,
    int refWidth, int refHeight, int targetWidth, int targetHeight, float* znccMat,
    size_t refPitch, size_t targetPitch, size_t depthPitch, size_t normalPitch, size_t outPitch);

// ============================================================================
// Host-side CUDA Setup Functions
// ============================================================================

/**
 * Copies camera data to device constant memory.
 *
 * @param h0_host     3x3 matrix H0 = K2 * R * K1^-1 (row-major)
 * @param k1inv_host  3x3 matrix K1^-1 (row-major)
 * @param e2_host     3-vector K2 * t
 */
void setupCameraMatrices(const float* h0_host, const float* k1inv_host, const float* e2_host);

/**
 * Sets the ZNCC window radius in device constant memory.
 *
 * @param radius    Window radius in pixels (e.g., 8 for 17x17 window)
 */
void setWindowRadius(int radius);

/**
 * Checks for CUDA errors and prints diagnostic information.
 */
void checkCudaError(const char* message);