#pragma once
#include <stdio.h>
#include <vector>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "simulationSettings.h"

struct deviceParticle
{
    // 0: x position
    // 1: y position
    // 2: z position
    // 3: ID
    // 4: x velocity
    // 5: y velocity
    // 6: z velocity
    // 7: mass
    // 8: smoothing length
    // 9: density
    // 10: omega
    // 11: pressure
    float particleData[PARTICLE_DATA_LENGTH];
};

struct particleSystem
{
    // Constructor/Destructor
    particleSystem();
    ~particleSystem();

    deviceParticle* h_deviceParticleList;

    float4* pos;
    float3* prevpos;
    float* mass;
    float* smoothingLengths;
    float* densities;
    float* omegas;
    float* pressures;

    // Initiates position, velocity, and copies memory to GPU
    void init();
    // Pulls data from GPU, "undoes" morton curve arrangement
    void getFromGPU(deviceParticle* d_deviceParticleList);
};

// Thrust functor to generate random particle positions
struct randPosFunctor
{
    __device__ float operator()(int idx);
};
