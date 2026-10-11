#pragma once
#include <cuda_runtime.h> // just for float3... meh
#include <array>

struct NormalPermutation
{
    float tangent;
    float bitangent;
};

struct FrontierPoint
{
    int x;
    int y;
    float depth;   // from neighbor parent
    float3 normal; // from neighbor parent
};

struct ResultPoint
{
    int x;
    int y;
    float depth;
    float3 normal;
	float znccScore = -1.f;
};

struct HomographyParams {
    std::array<float, 9> H0;     // K2 * R * K1^-1
    std::array<float, 9> K1inv;  // K1^-1
    std::array<float, 3> e2;     // K2 * t
};