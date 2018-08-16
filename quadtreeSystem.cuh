#pragma once
#include <vector>
#include "quadRectangle.cuh"
#include "particleSystem.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include "simulationSettings.h"

struct deviceQuad
{
    float hCell;
    int neibBucketCount;
    int* d_neibBucketsIndices;
    int firstContainedParticleIdx;
    // Note that this cannot exceed MAX_PARTICLES_PER_BUCKET
    int containedParticleCount;
};

struct quad
{
    // Constructor/Destructor
    quad();
    ~quad();

    // Quad data
    int index;
    float hCell = std::max(WIDTH, HEIGHT);
    int childrenIndices[4];
    int bucketIndex = 0;
    bool isBucket = true;

    std::vector<int> containedParticlesIndices;
    std::vector<int> neibBucketIndices;

    // Geometric data
    quadRectangle quadRect;

    // Subdivides into child quadrants
    void divide(std::vector<quad>& quadList, particleSystem& pSystem, int& bucketCounter);
    // Finds all bucket quads where distance < H_CELL
    void neibSearchTraversal(std::vector<quad>& quadList, int currentIndex);
};

struct quadtreeSystem
{
    // Constructor/Destructor
    quadtreeSystem();
    ~quadtreeSystem();

    std::vector<quad> quadList;
    deviceQuad* h_quadList;

    int quadCount;
    int bucketCounter = 0;

    // Reset all tree data, allocate a fixed amount of space for tree creation
    void reset();
    // Create tree
    void makeQuadtree(particleSystem& pSystem);
    // Find neighbour buckets for all bucket quads. Complexity is O(N*log(N))
    void findAllBucketNeibs();
    // Remove all non-bucket quads, in preparation to send the buckets to the device
    void eliminateBranches();
    // Neighbour finding kerel. Structured similarly to a CUDA device function, however it requires std::vectors for now and thus must be a CPU function
    void kernelFindNeibs_CPU(int blockIdx, int threadIdx, particleSystem& pSystem, std::vector<quad>& quadList);
    // Finds neighbouring particles to each particle.
    void findNeibParticles(particleSystem& pSystem);
    // Arranges the particles in pSystem into a morton curve to send to GPU
    void arrangeAndTransfer(particleSystem& pSystem, deviceQuad* d_quadList, deviceParticle* d_deviceParticleList);
    // Frees deviceQuad data on the GPU
    void getFromGPU(deviceQuad* d_quadList);
};
