#pragma once
#include "particleSystem.cuh"
#include "octreeSystem.cuh"

struct physicsSystemGPU
{
    // Constructor/Destructor
    physicsSystemGPU();
    ~physicsSystemGPU();

    // Computes densities for all particles on the GPU
    void computeDensity(octreeSystem& octSystem, particleSystem& pSystem, deviceOctant* d_octantList);
};

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double w(float r, float h);

// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double W(float r, float h);

// Derivative of W
__host__ __device__ double dWdh(float r, float h);

// Derivative of w
__host__ __device__ double dwdh(float r, float h);

// Density function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
__global__ void computeDensityKernel(deviceOctant* d_octantList, float4* d_pos, float* d_smoothingLengths, float* d_densities, float* d_mass);
