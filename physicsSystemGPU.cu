#include "physicsSystemGPU.cuh"
#define square(x) ((x) * (x))
#define cube(x) ((x) * (x) * (x))
#define area(x) (PI * square(x))
#define PI 3.14159f

// Constructor
physicsSystemGPU::physicsSystemGPU()
{

}

// Destructor
physicsSystemGPU::~physicsSystemGPU()
{

}

// Computes densities for all particles on the GPU
void physicsSystemGPU::computeDensity(octreeSystem& octSystem, particleSystem& pSystem, deviceOctant* d_octantList)
{
    dim3 blocksPerGrid(octSystem.octCount);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BUCKET);
    computeDensityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_octantList, pSystem.d_octs, pSystem.d_pos, pSystem.d_smoothingLengths, pSystem.d_densities, pSystem.d_mass);
}

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__device__ float w(float q)
{
    float sigma = 10.0f / (7.0f * PI);
    float piecewiseSpline = ((q < 1) ? (0.25f * cube(2 - q)) - cube(1 - q) : (q < 2) ? 0.25f * cube(2 - q) : 0);
    return sigma * piecewiseSpline;
}

// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__device__ float W(float r, float h)
{
    return w(r/h) / area(h);
}

// Density function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
__global__ void computeDensityKernel(deviceOctant* d_octantList, int* d_octs, float4* d_pos, float* d_smoothingLengths, float* d_densities, float* d_mass)
{
    deviceOctant oct = d_octantList[blockIdx.x];
    if (threadIdx.x >= oct.containedParticleCount)
        return;

    // Initialize data for this octant and this particle.
    int particleIdx = oct.d_containedParticleIndices[threadIdx.x];
    float smoothingLength = oct.neibSearchRadius; // d_smoothingLengths[particleIdx];
    float p = 0;

    // Use 512 bytes of shared mem to hold particle data within this oct
    __shared__ float4 containedParticles[32];
    float4 particlePos = d_pos[particleIdx];
    containedParticles[threadIdx.x] = particlePos;
    __syncthreads();

    for (int i = 0; i < oct.neibBucketCount; i++)
    {
        deviceOctant neibOct = d_octantList[oct.d_neibBucketsIndices[i]];

        // This loop causes bank conflicts that slow down cache access
        // TODO: Change how neibs are iterated through based on thread index
        for (int j = 0; j < neibOct.containedParticleCount; j++)
        {
            int neibIdx = neibOct.d_containedParticleIndices[j];

            float4 particlePos2;
            if (neibOct.d_containedParticleIndices = oct.d_containedParticleIndices)
                particlePos2 = containedParticles[j];
            else
                particlePos2 = d_pos[neibIdx];

            float neibMass = d_mass[neibIdx];
            float r = sqrt(square(particlePos2.x - particlePos.x) + square(particlePos2.y - particlePos.y));
            p += neibMass * W(r, smoothingLength);
        }
    }

    d_densities[particleIdx] = p;
}
