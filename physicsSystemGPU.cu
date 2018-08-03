#include "physicsSystemGPU.cuh"
#define square(x) ((x) * (x))
#define cube(x) ((x) * (x) * (x))
#define quartic(x) ((x) * (x) * (x) * (x))
#define PI 3.14159f
#define ETA 1.5f
#define THRESHOLD 0.001f

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
    computeDensityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_octantList, pSystem.d_pos, pSystem.d_smoothingLengths, pSystem.d_densities, pSystem.d_mass);
}

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double w(float r, float h)
{
    double q = r/h;
    double sigma = 10.0f / (7.0f * PI);
    double piecewiseSpline = ((q < 1) ? (0.25f * cube(2 - q)) - cube(1 - q) : (q < 2) ? 0.25f * cube(2 - q) : 0);
    return sigma * piecewiseSpline;
}

// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double W(float r, float h)
{
    return w(r, h) / square(h);
}

// Derivative of W
__host__ __device__ double dWdh(float r, float h)
{
    return (-2.0f / cube(h))*w(r, h) + (1.0f/square(h))*(dwdh(r, h));
}

// Derivative of w
__host__ __device__ double dwdh(float r, float h)
{
    double sigma = 10.0f / (7.0f * PI);
    double piecewiseDerivative = ((r/h) < 1) ? (-9*cube(r) + 12*h*square(r)) / (4*quartic(h)) : ((r/h) < 2) ? (3*r * square(2*h - r)) / (4*quartic(h)) : 0;
    return sigma * piecewiseDerivative;
}

// Density function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
__global__ void computeDensityKernel(deviceOctant* d_octantList, float4* d_pos, float* d_smoothingLengths, float* d_densities, float* d_mass)
{
    // Gather information about this thread's corresponding octant and particle.
    deviceOctant oct = d_octantList[blockIdx.x];
    if (threadIdx.x >= oct.containedParticleCount)
        return;
    int particleIdx = oct.d_containedParticleIndices[threadIdx.x];
    float h = d_smoothingLengths[particleIdx];

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
    double p, dpdh;
    Z = 1;

    // Calculate density at current smoothing length and adjust smoothing length using the Newton-Raphson method
    while (fabs(Z) > THRESHOLD && counter++ < 20)
    {
        p = 0;
        dpdh = 0;
        for (int i = 0; i < oct.neibBucketCount; i++)
        {
            deviceOctant neibOct = d_octantList[oct.d_neibBucketsIndices[i]];
            bool sameOct = neibOct.d_containedParticleIndices == oct.d_containedParticleIndices;

            for (int j = 0; j < neibOct.containedParticleCount; j++)
            {
                int neibIdx = neibOct.d_containedParticleIndices[j];
                float4 neibPos = sameOct ? containedParticles[j] : d_pos[neibIdx];
                float neibMass = sameOct ? containedParticles[j].z : d_mass[neibIdx];
                float r = sqrt(square(neibPos.x - particlePos.x) + square(neibPos.y - particlePos.y));
                dpdh += neibMass * dWdh(r, h);
                p += neibMass * W(r, h);
            }
        }

        // The Z function enforces the relationship in http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf
        Z = particleMass * square(ETA / h) - p;
        dZdh = (-2.0f * particleMass * square(ETA) / cube(h)) - dpdh;
        h -= (fabs(Z) > THRESHOLD) ? Z / dZdh : 0;
    }

    d_smoothingLengths[particleIdx] = h;
    d_densities[particleIdx] = p;
}
