#pragma once
#include "particleSystem.cuh"
#include "octreeSystem.cuh"

struct physicsSystemGPU
{
    // Constructor/Destructor
    physicsSystemGPU();
    ~physicsSystemGPU();

    void RunGPUSPH(octreeSystem& octSystem, deviceOctant* d_octantList, deviceParticle* d_deviceParticleList);
};

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double w(float r, float h);
// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double W(float r, float h);
// Derivatives
__host__ __device__ double dWdh(float r, float h);
__host__ __device__ double dWdr(float r, float h);
__host__ __device__ double dwdh(float r, float h);
__host__ __device__ double dwdr(float r, float h);

__global__ void SPHKernel(deviceOctant* d_octantList, deviceParticle* d_deviceParticleList);

 // Acceleration function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 30)
 /*__global__ void computeAccelerationKernel(deviceOctant* d_octantList,
                                           float4* d_pos,
                                           float3* d_vel,
                                           float* d_smoothingLengths,
                                           float* d_densities,
                                           float* d_mass,
                                           float* d_omegas,
                                           float* d_pressures,
                                           float dt);
*/
