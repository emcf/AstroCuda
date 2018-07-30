#include "physicsSystemGPU.cuh"
#define square(x) ((x) * (x))
#define cube(x) ((x) * (x) * (x))
#define quartic(x) ((x) * (x) * (x) * (x))
#define PI 3.14159f
#define ETA 1.9f
#define DELTA 0.001f

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
    computeDensityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_octantList, pSystem.d_octs, pSystem.d_pos, pSystem.d_smoothingLengths, pSystem.d_densities, pSystem.d_mass, pSystem.d_pressureGradients);
}

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__device__ double w(float r, float h)
{
    double q = r/h;
    double sigma = 10.0f / (7.0f * PI);
    double piecewiseSpline = ((q < 1) ? (0.25f * cube(2 - q)) - cube(1 - q) : (q < 2) ? 0.25f * cube(2 - q) : 0);
    return sigma * piecewiseSpline;
}

// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__device__ double W(float r, float h)
{
    return w(r, h) / square(h);
}

// Derivatives
__device__ double dWdh(float r, float h)
{
    return (-2.0f / cube(h))*w(r, h) + (1.0f/square(h))*(dwdh(r, h));
}

__device__ double dwdh(float r, float h)
{
    double sigma = 10.0f / (7.0f * PI);
    double q = r/h;
    double piecewiseDerivative = (q < 1) ? (-9.0f * cube(r) / (4.0f * quartic(h))) : (q < 2) ? (3.0f * cube(h) / (4.0f * quartic(h))) : 0.0f;
    return sigma * piecewiseDerivative;
}

// Density function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
__global__ void computeDensityKernel(deviceOctant* d_octantList, int* d_octs, float4* d_pos, float* d_smoothingLengths, float* d_densities, float* d_mass, float2* d_pressureGradients)
{
    // Gather information about this thread's corresponding octant and particle.
    deviceOctant oct = d_octantList[blockIdx.x];
    if (threadIdx.x >= oct.containedParticleCount)
        return;
    int particleIdx = oct.d_containedParticleIndices[threadIdx.x];
    float h = 10.0f;

    // Use MAX_PARTICLES_PER_BUCKET*4*4 bytes of shared mem to hold particle data within this oct
    // Each float4 holds {x, y, z, mass}
    __shared__ float4 containedParticles[MAX_PARTICLES_PER_BUCKET];
    float4 particlePos = d_pos[particleIdx];
    float particleMass = d_mass[particleIdx];
    containedParticles[threadIdx.x] = particlePos;
    containedParticles[threadIdx.x].z = particleMass;
    __syncthreads();

    int counter = 0;
    double Z, dZdh;
    double p, dpdh, dp;
    Z = 1;

    // Calculate density at current smoothing length and adjust smoothing length using the Newton-Raphson method
    while (fabs(Z) > 0.001f && counter++ < 100)
    {
        p = 0;
        //dpdh = 0;
        dp = 0;
        for (int i = 0; i < oct.neibBucketCount; i++)
        {
            deviceOctant neibOct = d_octantList[oct.d_neibBucketsIndices[i]];
            bool sameOct = neibOct.d_containedParticleIndices == oct.d_containedParticleIndices;
            for (int j = 0; j < neibOct.containedParticleCount; j++)
            {
                if (sameOct && threadIdx.x == j)
                    continue;
                int neibIdx = neibOct.d_containedParticleIndices[j];
                float4 neibPos = sameOct ? containedParticles[j] : d_pos[neibIdx];
                float neibMass = sameOct ? containedParticles[j].z : d_mass[neibIdx];
                float r = sqrt(square(neibPos.x - particlePos.x) + square(neibPos.y - particlePos.y));
                //dpdh += neibMass * dWdh(r, h);
                dp += neibMass * W(r, h + DELTA);
                p += neibMass * W(r, h);
            }
        }

        Z = particleMass * square(ETA / h) - p;
        dZdh = ((particleMass * square(ETA / (h + DELTA)) - dp) - (Z)) / (DELTA);
        //dZdh = (-2.0f * particleMass * square(ETA) / cube(h)) - dpdh;
        h -= Z / dZdh;
    }

    d_smoothingLengths[particleIdx] = h;
    d_densities[particleIdx] = p;
}
