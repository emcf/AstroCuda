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
    float neibSearchRadius;

    // Note that this cannot exceed MAX_PARTICLES_PER_BUCKET
    int containedParticleCount;
    int* d_containedParticleIndices;

    // Note that this cannot exceed MAX_NEIBS_PER_BUCKET
    int neibBucketCount;
    int* d_neibBucketsIndices;
};

struct octant
{
    // Constructor/Destructor
    octant();
    ~octant();

    // Octant data
    int index;
    int childrenIndices[4];
    int bucketIndex = 0;
    bool isBucket = true;

    std::vector<int> containedParticlesIndices;
    std::vector<int> neibBucketIndices;
    std::vector<int> neibParticleIndices;

    // SPH data
    float neibSearchRadius;

    // Geometric data
    octRectangle octRect;

    // Subdivides into child quadrants
    void divide(std::vector<octant>& octantList, particleSystem& pSystem, int& bucketCounter);
    // Finds all bucket octants where distance < H_CELL
    void findNeibBuckets(std::vector<octant>& octantList, int currentIndex);
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
    // Converts each bucket octant into a deviceOctant and sends it to the device.
    deviceOctant* sendToGPU();
    // Frees deviceOctant data on the GPU
    void freeFromGPU();
};