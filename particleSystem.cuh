#pragma once
#include <stdio.h>
#include <vector>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "simulationSettings.h"

struct particleSystem
{
    // Constructor/Destructor
    particleSystem();
    ~particleSystem();

    // Position
    // x, y, z, ID
    float4* pos;
    float4* d_pos;

    // Velocity
    // x, y, z
    float3* vel;
    float3* d_vel;

    float* mass;
    float* d_mass;

    float* smoothingLengths;
    float* d_smoothingLengths;

    float* densities;
    float* d_densities;

    float2* pressureGradients;
    float2* d_pressureGradients;

    // ID of containing octant
    int* octs;
    int* d_octs;

    // Initiates position, velocity, and copies memory to GPU
    void init();
    // Euler integration on particle positions using CUDA kernel
    void integrate();

    float getSquaredDistance(int indexA, int indexB);
};

// Thrust functor to generate random particle positions
struct randPosFunctor
{
    __device__ float operator()(int idx);
};
