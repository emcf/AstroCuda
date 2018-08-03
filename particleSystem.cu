#include "particleSystem.cuh"
#include "simulationSettings.h"

// Constructor
particleSystem::particleSystem()
{
    // Allocate memory for particles on host
    pos = (float4*)malloc(N*sizeof(float4));
    vel = (float3*)malloc(N*sizeof(float3));
    mass = (float*)malloc(N*sizeof(float));
    smoothingLengths = (float*)malloc(N*sizeof(float));
    densities = (float*)malloc(N*sizeof(float));

    // Allocate memory for particles on device
    cudaMalloc((void**) &d_pos, N*sizeof(float4));
    cudaMalloc((void**) &d_vel, N*sizeof(float3));
    cudaMalloc((void**) &d_mass, N*sizeof(float));
    cudaMalloc((void**) &d_smoothingLengths, N*sizeof(float));
    cudaMalloc((void**) &d_densities, N*sizeof(float));
}

// Destructor
particleSystem::~particleSystem()
{
    // Free host memory
    free(pos);
    free(vel);
    free(mass);
    free(smoothingLengths);
    free(densities);

    // Free device memory
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_mass);
    cudaFree(d_smoothingLengths);
    cudaFree(d_densities);
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
        vel[i].x = 0.0f; // h_randomFloats[3 * N + i] - 0.5f;
        vel[i].y = 0.0f; // h_randomFloats[4 * N + i] - 0.5f;
        smoothingLengths[i] = 20.0f;
    }

    cudaMemcpy(d_pos, pos, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_smoothingLengths, smoothingLengths, N*sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void kernelIntegrate(float4* pos, float3* vel, float dt)
{
    // Get index of threads. Corresponds directly to the particles being processed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N || i == j)
        return;

    float4 particle_i = pos[i];
    float4 particle_j = pos[j];

    // Get vector from i to j
    float2 r;
    r.x = particle_j.x - particle_i.x;
    r.y = particle_j.y - particle_i.y;
    float r_dot = (r.x * r.x + r.y * r.y) + GRAVITY_SOFTENING;

    // Calculate the gravitational force
    float force;
    force = G_CONST / r_dot;
    r.x *= force;
    r.y *= force;

    // Integrate velocity
    float3 particleVel_i = vel[i];
    float deltaVx = r.x * dt;
    float deltaVy = r.y * dt;
    particleVel_i.x += deltaVx;
    particleVel_i.y += deltaVy;

    // Invert velocities if the particle is headed towards a boundary
    if (particle_i.x + particleVel_i.x * dt <= 0 || particle_i.x + particleVel_i.x * dt >= WIDTH)
        particleVel_i.x *= -1;

    if (particle_i.y + particleVel_i.y * dt <= 0 || particle_i.y + particleVel_i.y * dt >= HEIGHT)
        particleVel_i.y *= -1;

    // Integrate position
    particle_i.x += particleVel_i.x * dt;
    particle_i.y += particleVel_i.y * dt;

    vel[i] = particleVel_i;
    pos[i] = particle_i;
}

void particleSystem::integrate()
{
    float totalBlocks = N * N / 1024;
    dim3 blocksPerGrid(sqrt(totalBlocks), sqrt(totalBlocks));
    dim3 threadsPerBlock(32, 32);
    // Run CUDA kernel
    kernelIntegrate<<<blocksPerGrid, threadsPerBlock>>>(d_pos, d_vel, DELTA_TIME);
    // Copy the integrated particles from the device to the host
    cudaMemcpy(pos, d_pos, N*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel, d_vel, N*sizeof(float3), cudaMemcpyDeviceToHost);
}

// Thrust functor to generate random particle positions
__device__ float randPosFunctor::operator()(int idx)
{
    thrust::default_random_engine randomEng;
    thrust::uniform_real_distribution<float> uniformRealDist;
    randomEng.discard(idx);
    // By multiplying two uniformly random real numbers, the results exponentially lean towards closer to 0 in the domain [0, 1]
    // This is ideal, since a non-uniform distribution represents astrophysical problems well.
    return uniformRealDist(randomEng) * ((UNIFORM_DISTRIBUTION) ? 1 : uniformRealDist(randomEng));
}
