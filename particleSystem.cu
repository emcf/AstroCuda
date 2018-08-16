#include "particleSystem.cuh"
#include "simulationSettings.h"

// Constructor
particleSystem::particleSystem()
{
    pos = (float4*)malloc(N*sizeof(float4));
    vel = (float3*)malloc(N*sizeof(float3));
    mass = (float*)malloc(N*sizeof(float));
    smoothingLengths = (float*)malloc(N*sizeof(float));
    densities = (float*)malloc(N*sizeof(float));
    pressures = (float*)malloc(N*sizeof(float));
}

// Destructor
particleSystem::~particleSystem()
{
    free(pos);
    free(vel);
    free(mass);
    free(smoothingLengths);
    free(densities);
    free(pressures);
}

void particleSystem::init()
{
    // Use thrust to generate an array of random floats, of length N
    thrust::counting_iterator<int> counter(0);
    thrust::device_vector<float> d_randomFloats(N * 6);
    thrust::transform(counter, counter + N * 6 , d_randomFloats.begin(), randPosFunctor());
    thrust::host_vector<float> h_randomFloats = d_randomFloats;

    for (int i = 0; i < N; i++)
    {
        pos[i].x = h_randomFloats[i] * WIDTH;
        pos[i].y = h_randomFloats[N + i] * HEIGHT;
        mass[i] = 3 + (h_randomFloats[2 * N + i] * 4.0f);
        vel[i].x = 0;
        vel[i].y = 0;
        smoothingLengths[i] = 20.0f;
    }
}

void particleSystem::getFromGPU(deviceParticle* d_deviceParticleList)
{
    cudaMemcpy(h_deviceParticleList, d_deviceParticleList, N*sizeof(deviceParticle), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        deviceParticle h_particle = h_deviceParticleList[i];
        int particleIdx = h_particle.particleData[3];
        pos[particleIdx].x = h_particle.particleData[0];
        pos[particleIdx].y = h_particle.particleData[1];
        pos[particleIdx].z = h_particle.particleData[2];
        vel[particleIdx].x = h_particle.particleData[4];
        vel[particleIdx].y = h_particle.particleData[5];
        vel[particleIdx].z = h_particle.particleData[6];
        mass[particleIdx] = h_particle.particleData[7];
        smoothingLengths[particleIdx] = h_particle.particleData[8];
        densities[particleIdx] = h_particle.particleData[9];
        pressures[particleIdx] = h_particle.particleData[10];
    }
}

// Thrust functor to generate random particle positions
__device__ float randPosFunctor::operator()(int idx)
{
    thrust::default_random_engine randomEng;
    thrust::uniform_real_distribution<float> uniformRealDist;
    randomEng.discard(idx);
    // By multiplying two uniformly random real numbers, the results exponentially lean towards closer to 0 in the domain [0, 1]
    // This is ideal, since a non-uniform distribution represents astrophysical problems well.
    return uniformRealDist(randomEng) * ((UNIFORM_DISTRIBUTION) ? 1 : (uniformRealDist(randomEng)));
}
