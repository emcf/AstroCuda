#pragma once
#include "particleSystem.cuh"
#include "quadtreeSystem.cuh"

struct physicsSystemGPU
{
    // N*N array holding gradients for W_i_j weight functions
    float2* gradW;
    float2* d_gradW;

    // Constructor/Destructor
    physicsSystemGPU();
    ~physicsSystemGPU();

    void solveSPH(quadtreeSystem& quadSystem, deviceQuad* d_quadList, deviceParticle* d_deviceParticleList);
    void integrate(quadtreeSystem& quadSystem, deviceQuad* d_quadList, deviceParticle* d_deviceParticleList);
};

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double w(float r, float h);
// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double W(float r, float h);
// Pressure function, https://arxiv.org/pdf/1007.1245.pdf (3.36)
__host__ __device__ double P(double p);
// Partial derivatives
__host__ __device__ double dWdh(float r, float h);
__host__ __device__ double dWdr(float r, float h);
__host__ __device__ double dwdh(float r, float h);
__host__ __device__ double dwdr(float r, float h);
__host__ __device__ double dwdq(float q);
// GPU Kernels
__global__ void SPHSolverKernel(deviceQuad* d_quadList, deviceParticle* d_deviceParticleList, float2* d_gradW);
__global__ void integratorKernel(deviceQuad* d_quadList, deviceParticle* d_deviceParticleList, float2* d_gradW);
