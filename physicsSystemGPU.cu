#include "physicsSystemGPU.cuh"
#define square(x) ((x) * (x))
#define cube(x) ((x) * (x) * (x))
#define quartic(x) ((x) * (x) * (x) * (x))
#define PI 3.14159f
#define ETA 2.0f
#define SIGMA (10.0f / (7.0f * PI))
// A large epsilon value simulates large distances between objects, akin to objects in space
#define EPSILON 100.0f
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
void physicsSystemGPU::computeDensity(octreeSystem& octSystem, particleSystem& pSystem, deviceOctant* d_octantList, deviceParticle* d_deviceParticleList)
{
    dim3 blocksPerGrid(octSystem.octCount);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BUCKET);
    computeDensityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_octantList, d_deviceParticleList);
}

void physicsSystemGPU::computeAcceleration(octreeSystem& octSystem, particleSystem& pSystem, deviceOctant* d_octantList)
{
    dim3 blocksPerGrid(octSystem.octCount);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BUCKET);
    //computeAccelerationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_octantList, pSystem.d_pos, pSystem.d_vel, pSystem.d_smoothingLengths, pSystem.d_densities, pSystem.d_mass, pSystem.d_omegas, pSystem.d_pressures, DELTA_TIME);
}

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double w(float r, float h)
{
    double q = r/h;
    double sigma = 10.0f / (7.0f * PI);
    double piecewiseSpline = (q < 1) ? (0.25f * cube(2 - q)) - cube(1 - q) :
                             (q < 2) ? 0.25f * cube(2 - q) :
                             0;
    return sigma * piecewiseSpline;
}

// Weight function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double W(float r, float h)
{
    return w(r, h) / square(h);
}

__host__ __device__ double dWdh(float r, float h)
{
    return (-2.0f / cube(h))*w(r, h) + (1.0f/square(h))*(dwdh(r, h));
}

__host__ __device__ double dWdr(float r, float h)
{
    return dwdr(r, h) / square(h);
}

__host__ __device__ double dwdh(float r, float h)
{
    double piecewiseDerivative = ((r/h) < 1) ? (-9*cube(r) + 12*h*square(r)) / (4*quartic(h)) :
                                 ((r/h) < 2) ? (3*r * square(2*h - r)) / (4*quartic(h)) :
                                 0;
    return SIGMA * piecewiseDerivative;
}

__host__ __device__ double dwdr(float r, float h)
{
    double piecewiseDerivative = ((r/h) < 1) ? (3*square(1-(r/h)) / h) - (3*square(2 - (r/h))/(4*h)) :
                                 ((r/h) < 2) ?  3*square(2-(r/h)) / (4*h) :
                                 0;
    return SIGMA * piecewiseDerivative;
}

__host__ __device__ double dhdp(float h, float p)
{
    return -h/(2*p);
}

// Density function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
__global__ void computeDensityKernel(deviceOctant* d_octantList, deviceParticle* d_deviceParticleList)
{
    // Gather basic octant/particle data
    deviceOctant oct = d_octantList[blockIdx.x];
    if (threadIdx.x >= oct.containedParticleCount)
        return;
    int particleIdx = oct.firstContainedParticleIdx + threadIdx.x;

    // Place this particle into shared memory cache, so other threads in this block can use it
    __shared__ deviceParticle containedParticles[MAX_PARTICLES_PER_BUCKET];
    deviceParticle thisParticle = d_deviceParticleList[particleIdx];
    containedParticles[threadIdx.x] = thisParticle;
    __syncthreads();

    int counter = 0;
    double h  = thisParticle.particleData[8];
    double mass = thisParticle.particleData[7];
    double Z = 1, dZdh, p, dpdh, omega, omegaSum;
    float2 a;

    // Calculate density at current smoothing length and adjust smoothing length using the Newton-Raphson method
    while ((fabs(Z) > THRESHOLD || h <= 0) && counter++ < 20)
    {
        // Reset SPH data
        p = 0;
        dpdh = 0;
        omegaSum = 0;
        a = {0, 0};

        // Iterate through each neib bucket, then iterate through each particle in that bucket
        for (int i = 0; i < oct.neibBucketCount; i++)
        {
            deviceOctant neibOct = d_octantList[oct.d_neibBucketsIndices[i]];
            bool sameOct = neibOct.firstContainedParticleIdx == oct.firstContainedParticleIdx;
            for (int j = 0; j < neibOct.containedParticleCount; j++)
            {
                int neibIdx = neibOct.firstContainedParticleIdx + j;
                // If the neib particle is within this octant, ensure shared memory cache is used
                deviceParticle neibParticle = sameOct ? containedParticles[j] : d_deviceParticleList[neibIdx];
                // Calculate SPH things
                float neibMass = neibParticle.particleData[7];
                float rx = neibParticle.particleData[0] - thisParticle.particleData[0];
                float ry = neibParticle.particleData[1] - thisParticle.particleData[1];
                float r = sqrt(square(rx) + square(ry) + EPSILON);
                // Increment acceleration
                a.x += (rx/r) * (G_CONST * neibMass / square(r));
                a.y += (ry/r) * (G_CONST * neibMass / square(r));
                // Increment density and density deritative
                dpdh += neibMass * dWdh(r, h);
                p += neibMass * W(r, h);
                // Intrement a sum for further use in Omega calculation
                omegaSum += neibMass * dWdh(r, h);
            }
        }

        // Using Newton-Raphson method on this Z calculation enforces the relationship in http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf
        Z = mass * square(ETA / h) - p;
        dZdh = (-2.0f * mass * square(ETA) / cube(h)) - dpdh;
        h -= Z / dZdh;
        omega = 1.0f - (dhdp(h, p) * omegaSum);
    }

    // Verlet integration
    float2 prevPos = {thisParticle.particleData[4], thisParticle.particleData[5]};
    float2 pos = {thisParticle.particleData[0], thisParticle.particleData[1]};
    float2 newPos = {pos.x * 2 - prevPos.x + a.x * square(DELTA_TIME),
                     pos.y * 2 - prevPos.y + a.y * square(DELTA_TIME)};

    // Assign new integrated position
    d_deviceParticleList[particleIdx].particleData[0] = newPos.x;
    d_deviceParticleList[particleIdx].particleData[1] = newPos.y;
    d_deviceParticleList[particleIdx].particleData[4] = pos.x;
    d_deviceParticleList[particleIdx].particleData[5] = pos.y;

    // Assign SPH data
    d_deviceParticleList[particleIdx].particleData[8] = h;
    d_deviceParticleList[particleIdx].particleData[9] = p;
    d_deviceParticleList[particleIdx].particleData[10] = omega;
}

/*
// Acceleration function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 30)
__global__ void computeAccelerationKernel(deviceOctant* d_octantList,
                                          float4* d_pos,
                                          float3* d_vel,
                                          float* d_smoothingLengths,
                                          float* d_densities,
                                          float* d_mass,
                                          float* d_omegas,
                                          float* d_pressures,
                                          float dt)
{
    // Gather information about this thread's corresponding octant and particle.
    deviceOctant oct = d_octantList[blockIdx.x];
    if (threadIdx.x >= oct.containedParticleCount)
        return;
    int particleIdx = oct.d_containedParticleIndices[threadIdx.x];
    double h = d_smoothingLengths[particleIdx];

    // Use MAX_PARTICLES_PER_BUCKET*4*4 bytes of shared mem to hold particle data within this oct
    // Each float4 holds {x, y, z, mass}
    __shared__ float4 containedParticles[MAX_PARTICLES_PER_BUCKET];
    float4 particlePos = d_pos[particleIdx];
    float3 particleVel = d_vel[particleIdx];
    float particleMass = d_mass[particleIdx];
    float particleOmega = d_omegas[particleIdx];
    float particlePressure = d_pressures[particleIdx];
    float particleDensity = d_densities[particleIdx];
    containedParticles[threadIdx.x] = particlePos;
    containedParticles[threadIdx.x].z = particleMass;
    __syncthreads();

    float accel = 0;
    for (int i = 0; i < oct.neibBucketCount; i++)
    {
        deviceOctant neibOct = d_octantList[oct.d_neibBucketsIndices[i]];
        bool sameOct = neibOct.d_containedParticleIndices == oct.d_containedParticleIndices;

        for (int j = 0; j < neibOct.containedParticleCount; j++)
        {
            int neibIdx = neibOct.d_containedParticleIndices[j];
            float4 neibPos = sameOct ? containedParticles[j] : d_pos[neibIdx];
            float neibMass = sameOct ? containedParticles[j].z : d_mass[neibIdx];
            float neibSmoothingLength =  d_smoothingLengths[neibIdx];
            float neibOmega = d_omegas[neibIdx];
            float neibPressure = 1; //d_pressures[neibIdx];
            float neibDensity  = d_densities[neibIdx];
            float r = sqrt(square(neibPos.x - particlePos.x) + square(neibPos.y - particlePos.y));

            // http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 30)
            float termA = ((particlePressure / (particleOmega * square(particleDensity))) * dWdr(r, h));
            float termB = ((neibPressure / (neibOmega * square(neibDensity))) * dWdr(r, neibSmoothingLength));
            accel += neibMass * (termA - termB);
        }
    }

    accel *= -0.001;
    d_vel[particleIdx].x = particleVel.x + accel * dt;
}*/
