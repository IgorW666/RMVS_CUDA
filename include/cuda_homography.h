#pragma once

#include <cuda_runtime.h>
#include "types.h"

/**
 * Evaluates ZNCC (Zero-mean Normalized Cross-Correlation) for depth map
 * using plane-induced homography and bicubic interpolation.
 *
 * @param refWidth/refHeight        Reference image size (also depth/normal/output size)
 * @param targetWidth/targetHeight  Target image size
 */
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
    int frontierCount); // <-- added frontierCount parameter

/**
 * Dual-target launcher: computes ZNCC against two different target images in one kernel pass.
 *
 * result0 and result1 must be allocated with size frontierCount * depthNSteps * normalNSteps.
 */
void launchEvaluateZNCC_DualTargets(
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
    int target1Height,
    int frontierCount);

/**
 * Dual-target evaluate kernel (device code).
 */
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
    int target1Height);

/**
 * Copies camera data to device constant memory.
 *
 * @param h0_host     3x3 matrix H0 = K2 * R * K1^-1 (row-major)
 * @param k1inv_host  3x3 matrix K1^-1 (row-major)
 * @param e2_host     3-vector K2 * t
 */
void setupCameraMatrices(const float* h0_host, const float* k1inv_host, const float* e2_host);

/**
 * Copies second-camera data to device constant memory (non-breaking addition).
 */
void setupCameraMatricesSecond(const float* h1_host, const float* k1inv1_host, const float* e2_1_host);

void setupNormalPermutations(const NormalPermutation* normalPermutations, int count);

/**
 * Sets the ZNCC window radius in device constant memory.
 *
 * @param radius    Window radius in pixels (e.g., 8 for 17x17 window)
 */
void setWindowRadius(int radius);

/**
 * Set image row-pitches (size_t) in device constant memory.
 * Call this from host after creating / uploading device images.
 *
 * @param refPitch_host    pitch (bytes) of the reference image rows on device
 * @param targetPitch_host pitch (bytes) of the target image rows on device
 */
void setImagePitches(size_t refPitch_host, size_t targetPitch_host);

/**
 * Set image row-pitches for dual-target kernel (non-breaking addition).
 */
void setImagePitchesDual(size_t refPitch_host, size_t targetPitch_host, size_t target2Pitch_host);

/**
 * Checks for CUDA errors and prints diagnostic information.
 */
void checkCudaError(const char* message);

// unused single cam kernel
/**
 * Evaluates ZNCC kernel (device code).
 */
 /*__global__
 void evaluateZNCC(const float* refImg,
     const float* targetImg,
     const FrontierPoint* frontier,   // array with frontier points with depth and normal inherited from parent but new {x,y} <int>
     const float* depthSteps,         // array with depth values for that frontier point eg {-0.98, -0.99, 1.00, 1.01, 1.02 } but 8 or 16 values
     float* result,
     int depthNSteps,
     int normalNSteps,
     int refWidth,
     int refHeight,
     int targetWidth,
     int targetHeight);*/