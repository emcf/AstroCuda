#pragma once
#include <vector>
#include "octRectangle.cuh"
#include "particleSystem.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include "simulationSettings.h"

struct deviceOctant
{
    float hCell;
    int neibBucketCount;
    int* d_neibBucketsIndices;
    int firstContainedParticleIdx;
    // Note that this cannot exceed MAX_PARTICLES_PER_BUCKET
    int containedParticleCount;
};

struct octant
{
    // Constructor/Destructor
    octant();
    ~octant();

    // Octant data
    int index;
    float hCell = std::max(WIDTH, HEIGHT);
    int childrenIndices[4];
    int bucketIndex = 0;
    bool isBucket = true;

    std::vector<int> containedParticlesIndices;
    std::vector<int> neibBucketIndices;

    // Geometric data
    octRectangle octRect;

    // Subdivides into child quadrants
    void divide(std::vector<octant>& octantList, particleSystem& pSystem, int& bucketCounter);
    // Finds all bucket octants where distance < H_CELL
    void neibSearchTraversal(std::vector<octant>& octantList, int currentIndex);
};

struct octreeSystem
{
    // Constructor/Destructor
    octreeSystem();
    ~octreeSystem();

    std::vector<octant> octantList;
    deviceOctant* h_octantList;

    int octCount;
    int bucketCounter = 0;

    // Reset all tree data, allocate a fixed amount of space for tree creation
    void reset();
    // Create tree
    void makeOctree(particleSystem& pSystem);
    // Find neighbour buckets for all bucket octants. Complexity is O(N*log(N))
    void findAllBucketNeibs();
    // Remove all non-bucket octants, in preparation to send the buckets to the device
    void eliminateBranches();
    // Neighbour finding kerel. Structured similarly to a CUDA device function, however it requires std::vectors for now and thus must be a CPU function
    void kernelFindNeibs_CPU(int blockIdx, int threadIdx, particleSystem& pSystem, std::vector<octant>& octantList);
    // Finds neighbouring particles to each particle.
    void findNeibParticles(particleSystem& pSystem);
    // Arranges the particles in pSystem into a morton curve to send to GPU
    void arrangeAndTransfer(particleSystem& pSystem, deviceOctant* d_octantList, deviceParticle* d_deviceParticleList);
    // Frees deviceOctant data on the GPU
    void getFromGPU(deviceOctant* d_octantList);
};
