#pragma once

#include "image_handler.h"
#include <cuda_runtime.h>

class ZNCCProcessor {
public:
    ZNCCProcessor(int imgWidth, int imgHeight, int windowRadius);
    ~ZNCCProcessor();

    /**
 * Set camera data.
 *
 * @param H0    3x3 matrix K2 * R * K1^-1 (row-major)
 * @param K1inv 3x3 matrix K1^-1 (row-major)
 * @param e2    3-vector K2 * t
 */
    void setCameraMatrices(const float* H0, const float* K1inv, const float* e2);
    

    /**
     * Execute ZNCC computation.
     *
     * @param refImage      Reference image (GPU memory)
     * @param targetImage   Target image (GPU memory)
     * @param depthMap      Depth map (GPU memory)
     * @param normalMap     Normal map (GPU memory)
     * @return              Result ZNCC scores (GPU memory)
     */
    ImageData computeZNCC(const ImageData& refImage, const ImageData& targetImage,
        const ImageData& depthMap, const ImageData& normalMap);

    /**
     * Get optimal block/grid dimensions for current image size.
     */
    struct LaunchConfig {
        dim3 blockDim;
        dim3 gridDim;
    };

    LaunchConfig getOptimalLaunchConfig() const;

private:
    int imgWidth;
    int imgHeight;
    int windowRadius;
    ImageData znccOutput;
};