#pragma once
#include <cuda_runtime.h> // just for float3... meh

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

//struct DepthStep
//{
//	float depthMult; // depth multiplier for the current step    
//};