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
#define VISCOSITY 1.0f

// Constructor
physicsSystemGPU::physicsSystemGPU()
{
    gradW = (float2*)malloc(N*N*sizeof(float2));
    cudaMalloc((void**) &d_gradW, N*N*sizeof(float2));
}

// Destructor
physicsSystemGPU::~physicsSystemGPU()
{
    free(gradW);
    cudaFree(d_gradW);
}

// Computes densities for all particles on the GPU
void physicsSystemGPU::solveSPH(quadtreeSystem& quadSystem, deviceQuad* d_quadList, deviceParticle* d_deviceParticleList)
{
    // Allocate a block for each quad, and a thread for each particle within
    dim3 blocksPerGrid(quadSystem.quadCount);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BUCKET);
    SPHSolverKernel<<<blocksPerGrid, threadsPerBlock>>>(d_quadList, d_deviceParticleList, d_gradW);
}

void physicsSystemGPU::integrate(quadtreeSystem& quadSystem, deviceQuad* d_quadList, deviceParticle* d_deviceParticleList)
{
    // Allocate a block for each quad, and a thread for each particle within
    dim3 blocksPerGrid(quadSystem.quadCount);
    dim3 threadsPerBlock(MAX_PARTICLES_PER_BUCKET);
    integratorKernel<<<blocksPerGrid, threadsPerBlock>>>(d_quadList, d_deviceParticleList, d_gradW);
}

// M4 Cubic Spline smoothing kernel. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 6)
__host__ __device__ double w(float r, float h)
{
    double q = r/h;
    double piecewiseSpline = (q < 1) ? (0.25f * cube(2 - q)) - cube(1 - q) :
                             (q < 2) ? 0.25f * cube(2 - q) :
                             0.0f;
    return SIGMA * piecewiseSpline;
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
    double piecewiseDerivative = ((r/h) < 1) ? (-9.0f*cube(r) + 12.0f*h*square(r)) / (4.0f*quartic(h)) :
                                 ((r/h) < 2) ? (3.0f*r * square(2.0f*h - r)) / (4.0f*quartic(h)) :
                                 0.0f;
    return SIGMA * piecewiseDerivative;
}

__host__ __device__ double dwdr(float r, float h)
{
    double piecewiseDerivative = ((r/h) < 1) ? (3.0f*square(1.0f-(r/h)) / h) - (3.0f*square(2.0f - (r/h))/(4.0f*h)) :
                                 ((r/h) < 2) ?  3.0f*square(2.0f-(r/h)) / (4.0f*h) :
                                 0.0f;
    return SIGMA * piecewiseDerivative;
}

__host__ __device__ double dwdq(float q)
{
    double piecewiseDerivative = (q < 1) ? (3.0f*q*(3.0f*q-4.0f))/4.0f :
                                 (q < 2) ? -(3.0f*square(q-2.0f))/4.0f :
                                 0.0f;
    return SIGMA * piecewiseDerivative;
}

__host__ __device__ double dhdp(float h, float p)
{
    return -h/(2*p);
}

// Pressure function, https://arxiv.org/pdf/1007.1245.pdf (3.36)
__host__ __device__ double P(double p)
{
    #define INTERNAL_ENERGY 1.0f
    return ((5.0f/3.0f) - 1) * p * INTERNAL_ENERGY;
}

__global__ void SPHSolverKernel(deviceQuad* d_quadList, deviceParticle* d_deviceParticleList, float2* d_gradW)
{
    // Gather basic quad/particle data
    deviceQuad quad = d_quadList[blockIdx.x];
    int particleIdx = quad.firstContainedParticleIdx + threadIdx.x;
    if (threadIdx.x >= quad.containedParticleCount)
        return;

    // Place this particle into shared memory cache, so other threads in this block can use it
    __shared__ deviceParticle containedParticles[MAX_PARTICLES_PER_BUCKET];
    deviceParticle thisParticle = d_deviceParticleList[particleIdx];
    containedParticles[threadIdx.x] = thisParticle;
    __syncthreads();

    // Declare registers
    int counter = 0;
    double mass = thisParticle.particleData[7];
    double h  = thisParticle.particleData[8];
    double Z = 1, dZdh, p, dpdh;

    // Calculate density/acceleration at current smoothing length,
    // Adjust smoothing length using the Newton-Raphson method
    while ((fabs(Z) > THRESHOLD || h <= 0) && counter++ < 3)
    {
        // Reset SPH data
        p = 0;
        dpdh = 0;

        // Iterate through each neib bucket
        for (int i = 0; i < quad.neibBucketCount; i++)
        {
            deviceQuad neibQuad = d_quadList[quad.d_neibBucketsIndices[i]];
            bool sameQuad = neibQuad.firstContainedParticleIdx == quad.firstContainedParticleIdx;
            // Iterate through each particle in neib bucket
            for (int j = 0; j < neibQuad.containedParticleCount; j++)
            {
                // Allows shared memory multicasts to occur
                __syncthreads();
                int neibIdx = neibQuad.firstContainedParticleIdx + j;
                // If the neib particle is within this quad, ensure shared memory cache is used
                deviceParticle neibParticle = sameQuad ? containedParticles[j] : d_deviceParticleList[neibIdx];
                // Calculate SPH things
                float neibMass = neibParticle.particleData[7];
                float rx = neibParticle.particleData[0] - thisParticle.particleData[0];
                float ry = neibParticle.particleData[1] - thisParticle.particleData[1];
                float r = sqrt(square(rx) + square(ry) + EPSILON);
                // Increment density and density deritative. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
                dpdh += neibMass * dWdh(r, h);
                p += neibMass * W(r, h);
                // Find kernel gradient https://academic.oup.com/mnras/article/471/2/2357/3906602 (Equation 4)
                float q = r/h;
                float tempGradWx = rx * ((1.0f / quartic(h)) * (1.0f/q) * dwdq(q));
                float tempGradWy = ry * ((1.0f / quartic(h)) * (1.0f/q) * dwdq(q));
                d_gradW[particleIdx * N + neibIdx] = {tempGradWx, tempGradWy};
            }
        }

        // Using Newton-Raphson method on this Z calculation enforces the relationship in http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 10)
        Z = mass * square(ETA / h) - p;
        dZdh = (-2.0f * mass * square(ETA) / cube(h)) - dpdh;
        h -= Z / dZdh;
    }

    // Assign SPH data
    d_deviceParticleList[particleIdx].particleData[8] = h;
    d_deviceParticleList[particleIdx].particleData[9] = p;
    d_deviceParticleList[particleIdx].particleData[10] = P(p);

    // Smoothing length of quad for the next iteration should
    // be roughly 2 times the max smoothing length in quad
    if (h > d_quadList[blockIdx.x].hCell)
        d_quadList[blockIdx.x].hCell = 2 * h;
}

__global__ void integratorKernel(deviceQuad* d_quadList, deviceParticle* d_deviceParticleList, float2* d_gradW)
{
    // Gather basic quad/particle data
    deviceQuad quad = d_quadList[blockIdx.x];
    int particleIdx = quad.firstContainedParticleIdx + threadIdx.x;
    if (threadIdx.x >= quad.containedParticleCount)
        return;

    // Place this particle into shared memory cache, so other threads in this block can use it
    __shared__ deviceParticle containedParticles[MAX_PARTICLES_PER_BUCKET];
    deviceParticle thisParticle = d_deviceParticleList[particleIdx];
    containedParticles[threadIdx.x] = thisParticle;
    __syncthreads();

    // Acceleration
    float2 dvdt = {0, 0};

    // Iterate through each neib bucket
    for (int i = 0; i < quad.neibBucketCount; i++)
    {
        deviceQuad neibQuad = d_quadList[quad.d_neibBucketsIndices[i]];
        bool sameQuad = neibQuad.firstContainedParticleIdx == quad.firstContainedParticleIdx;
        // Iterate through each particle in neib bucket
        for (int j = 0; j < neibQuad.containedParticleCount; j++)
        {
            // Allows shared memory multicasts to occur
            __syncthreads();
            int neibIdx = neibQuad.firstContainedParticleIdx + j;
            // If the neib particle is within this quad, ensure shared memory cache is used
            deviceParticle neibParticle = sameQuad ? containedParticles[j] : d_deviceParticleList[neibIdx];
            // Calculate SPH things
            float neibMass = neibParticle.particleData[7];
            // Increment acceleration https://academic.oup.com/mnras/article/471/2/2357/3906602 (Equation 12)
            float2 gradW_i_j = d_gradW[particleIdx * N + neibIdx];
            float thisDensity = thisParticle.particleData[9];
            float neibDensity = neibParticle.particleData[9];
            float thisPressure = thisParticle.particleData[10];
            float neibPressure = neibParticle.particleData[10];
            dvdt.x += neibMass * ((thisPressure + neibPressure) / (thisDensity * neibDensity) + VISCOSITY) * gradW_i_j.x;
            dvdt.y += neibMass * ((thisPressure + neibPressure) / (thisDensity * neibDensity) + VISCOSITY) * gradW_i_j.y;
        }
    }

    // Invert the sign of dvdt to match the equation
    dvdt.x *= -1.0f;
    dvdt.y *= -1.0f;

    // Integrate velocity in registers (for faster position integration) and in global mem
    thisParticle.particleData[4] += dvdt.x * DELTA_TIME;
    thisParticle.particleData[5] += dvdt.y * DELTA_TIME;
    d_deviceParticleList[particleIdx].particleData[4] = thisParticle.particleData[4];
    d_deviceParticleList[particleIdx].particleData[5] = thisParticle.particleData[5];

    // Integrate position in global mem
    d_deviceParticleList[particleIdx].particleData[0] = thisParticle.particleData[0] + (thisParticle.particleData[4] * DELTA_TIME);
    d_deviceParticleList[particleIdx].particleData[1] = thisParticle.particleData[1] + (thisParticle.particleData[5] * DELTA_TIME);
}

/*
// Acceleration function. http://users.monash.edu.au/~dprice/SPH/price-spmhd.pdf (Equation 30)
__global__ void computeAccelerationKernel(deviceQuad* d_quadList,
                                          float4* d_pos,
                                          float3* d_vel,
                                          float* d_smoothingLengths,
                                          float* d_densities,
                                          float* d_mass,
                                          float* d_omegas,
                                          float* d_pressures,
                                          float dt)
{
    // Gather information about this thread's corresponding quad and particle.
    deviceQuad quad = d_quadList[blockIdx.x];
    if (threadIdx.x >= quad.containedParticleCount)
        return;
    int particleIdx = quad.d_containedParticleIndices[threadIdx.x];
    double h = d_smoothingLengths[particleIdx];

    // Use MAX_PARTICLES_PER_BUCKET*4*4 bytes of shared mem to hold particle data within this quad
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
    for (int i = 0; i < quad.neibBucketCount; i++)
    {
        deviceQuad neibQuad = d_quadList[quad.d_neibBucketsIndices[i]];
        bool sameQuad = neibQuad.d_containedParticleIndices == quad.d_containedParticleIndices;

        for (int j = 0; j < neibQuad.containedParticleCount; j++)
        {
            int neibIdx = neibQuad.d_containedParticleIndices[j];
            float4 neibPos = sameQuad ? containedParticles[j] : d_pos[neibIdx];
            float neibMass = sameQuad ? containedParticles[j].z : d_mass[neibIdx];
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
