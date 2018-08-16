#pragma once
#include <stdio.h>
#include <vector>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "simulationSettings.h"

// Must satisfy the following:
// MAX_PARTICLES_PER_BUCKET * sizeof(deviceParticle) <= 49152
struct deviceParticle
{
    // 0: x position
    // 1: y position
    // 2: z position
    // 3: ID
    // 4: x vel
    // 5: y vel
    // 6: z vel
    // 7: mass
    // 8: smoothing length
    // 9: density
    // 10: pressure
    float particleData[PARTICLE_DATA_LENGTH];
};

struct particleSystem
{
    // Constructor/Destructor
    particleSystem();
    ~particleSystem();

    deviceParticle* h_deviceParticleList;

    float4* pos;
    float3* vel;
    float* mass;
    float* smoothingLengths;
    float* densities;
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
