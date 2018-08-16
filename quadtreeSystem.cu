#include "quadtreeSystem.cuh"
#include <stdio.h>

// Constructor
quadtreeSystem::quadtreeSystem()
{

}

// Destructor
quadtreeSystem::~quadtreeSystem()
{

}

void quadtreeSystem::reset()
{
    // initialize quadtreeSystem
    quadList.clear();
    bucketCounter = 0;

    // Initialize the parent quad's cell
    quadRectangle parentRect;
    parentRect.topLeft.x = 0;
    parentRect.topLeft.y = HEIGHT;
    parentRect.height = HEIGHT;
    parentRect.width = WIDTH;
    parentRect.calculateVertices();

    // Initialize the parent quad
    quad parentQuad;
    parentQuad.index = 0;
    parentQuad.quadRect = parentRect;
    parentQuad.isBucket = true;
    parentQuad.containedParticlesIndices.clear();

    // Reserve enough space for a fixed number of quads. This is required since dynamically allocating the
    // vector from a void inside one of the vector's contained objects will change memory addresses to the contained object,
    // The problem is that C++ will still use the old memory address for obtaining variables stored by that object.
    quadList.reserve(QUAD_LIST_ALLOC);
    quadList.push_back(parentQuad);
}

void quadtreeSystem::makeQuadtree(particleSystem& pSystem)
{
    // set parent quad's contained particles to 1..N
    for (int i = 0; i < N; i++)
    {
        bool withinWidth = (quadList[0].quadRect.topLeft.x < pSystem.pos[i].x && pSystem.pos[i].x < quadList[0].quadRect.topRight.x);
        bool withinHeight = (quadList[0].quadRect.bottomLeft.y < pSystem.pos[i].y && pSystem.pos[i].y < quadList[0].quadRect.topLeft.y);
        if (withinWidth && withinHeight)
            quadList[0].containedParticlesIndices.push_back(i);
    }

    // Divide the first quad
    if (quadList[0].containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
    {
        quadList[0].isBucket = false;
        quadList[0].divide(quadList, pSystem, bucketCounter);
    }
}

// Arranges the particles in pSystem into a morton curve to send to GPU
void quadtreeSystem::arrangeAndTransfer(particleSystem& pSystem, deviceQuad* d_quadList, deviceParticle* d_deviceParticleList)
{
    pSystem.h_deviceParticleList = new deviceParticle[N];
    h_quadList = new deviceQuad[quadCount];

    int MortonCounter = 0;
    for (int i = 0; i < quadCount; i++)
    {
        int containedParticlesCount = quadList[i].containedParticlesIndices.size();

        h_quadList[i].hCell = quadList[i].hCell;

        // Transfers info storing the # of contained particles as well as # of neib buckets for this bucket
        h_quadList[i].containedParticleCount = containedParticlesCount;
        h_quadList[i].neibBucketCount = quadList[i].neibBucketIndices.size();

        // Allocate and transfer contained particle indices to the GPU
        h_quadList[i].firstContainedParticleIdx = MortonCounter;

        // Allocate and transfer neib bucket indices to the GPU
        cudaMalloc((void**) &(h_quadList[i].d_neibBucketsIndices), h_quadList[i].neibBucketCount * sizeof(int));
        cudaMemcpy(h_quadList[i].d_neibBucketsIndices, &(quadList[i].neibBucketIndices[0]), h_quadList[i].neibBucketCount * sizeof(int), cudaMemcpyHostToDevice);

        for (int j = 0; j < containedParticlesCount; j++)
        {
            // Convert particle info into congruent arrays
            deviceParticle h_particle;
            int particleIdx = quadList[i].containedParticlesIndices[j];
            h_particle.particleData[0] = pSystem.pos[particleIdx].x;
            h_particle.particleData[1] = pSystem.pos[particleIdx].y;
            h_particle.particleData[2] = pSystem.pos[particleIdx].z;
            h_particle.particleData[3] = particleIdx;
            h_particle.particleData[4] = pSystem.vel[particleIdx].x;
            h_particle.particleData[5] = pSystem.vel[particleIdx].y;
            h_particle.particleData[6] = pSystem.vel[particleIdx].z;
            h_particle.particleData[7] = pSystem.mass[particleIdx];
            h_particle.particleData[8] = pSystem.smoothingLengths[particleIdx];
            h_particle.particleData[9] = pSystem.densities[particleIdx];
            h_particle.particleData[10] = pSystem.pressures[particleIdx];
            pSystem.h_deviceParticleList[MortonCounter] = h_particle;
            MortonCounter++;
        }
    }

    // QuadList data is on GPU, now transfer the "meta data" to the GPU
    cudaMemcpy(d_quadList, h_quadList, quadCount*sizeof(deviceQuad), cudaMemcpyHostToDevice);
    // Transfer rearranged particle data to GPU
    cudaMemcpy(d_deviceParticleList, pSystem.h_deviceParticleList, N*sizeof(deviceParticle), cudaMemcpyHostToDevice);
}

void quadtreeSystem::getFromGPU(deviceQuad* d_quadList)
{
    cudaMemcpy(h_quadList, d_quadList, quadCount*sizeof(deviceQuad), cudaMemcpyDeviceToHost);

    // Free dyanmically allocated memory
    for (int i = 0; i < quadCount; i++)
    {
        quadList[i].hCell = h_quadList[i].hCell;
        cudaFree(h_quadList[i].d_neibBucketsIndices);
    }

    delete[] h_quadList;
}


void quadtreeSystem::eliminateBranches()
{
    for (int i = 0; i < quadList.size(); i+=0)
    {
        if (quadList[i].isBucket)
            i++;
        else
            quadList.erase(quadList.begin() + i);
    }

    quadCount = quadList.size();
}

// Find neighbour buckets for all bucket quads. Complexity is O(N*log(N))
void quadtreeSystem::findAllBucketNeibs()
{
    for (int i = 0; i < quadList.size(); i++)
    {
        if (quadList[i].isBucket)
            quadList[i].neibSearchTraversal(quadList, 0);
    }
}

// Constructor
quad::quad()
{

}

// Destructor
quad::~quad()
{

}

void quad::divide(std::vector<quad>& quadList, particleSystem& pSystem, int& bucketCounter)
{
    // Top left
    quad topLeftChild;
    //topLeftChild.parentIndex = index;
    topLeftChild.quadRect.topLeft = quadRect.topLeft;
    topLeftChild.quadRect.width = quadRect.width / 2.00f;
    topLeftChild.quadRect.height = quadRect.height / 2.00f;
    topLeftChild.quadRect.calculateVertices();
    // Top right
    quad topRightChild;
    //topRightChild.parentIndex = index;
    topRightChild.quadRect.topLeft.x = quadRect.centre.x;
    topRightChild.quadRect.topLeft.y = quadRect.topRight.y;
    topRightChild.quadRect.width = quadRect.width / 2.00f;
    topRightChild.quadRect.height = quadRect.height / 2.00f;
    topRightChild.quadRect.calculateVertices();
    // Bottom left
    quad bottomLeftChild;
    //bottomLeftChild.parentIndex = index;
    bottomLeftChild.quadRect.topLeft.x = quadRect.topLeft.x;
    bottomLeftChild.quadRect.topLeft.y =  quadRect.centre.y;
    bottomLeftChild.quadRect.width = quadRect.width / 2.00f;
    bottomLeftChild.quadRect.height = quadRect.height / 2.00f;
    bottomLeftChild.quadRect.calculateVertices();
    // Bottom right
    quad bottomRightChild;
    //bottomRightChild.parentIndex = index;
    bottomRightChild.quadRect.topLeft = quadRect.centre;
    bottomRightChild.quadRect.width = quadRect.width / 2.00f;
    bottomRightChild.quadRect.height = quadRect.height / 2.00f;
    bottomRightChild.quadRect.calculateVertices();

    // Assign particles to each new child quad
    for (int i = 0; i < containedParticlesIndices.size(); i++)
    {
        bool withinTopHalf =  pSystem.pos[containedParticlesIndices[i]].y > quadRect.centre.y;
        bool withinLeftHalf =  pSystem.pos[containedParticlesIndices[i]].x < quadRect.centre.x;

        if (withinTopHalf && withinLeftHalf)
            topLeftChild.containedParticlesIndices.push_back(containedParticlesIndices[i]);
        else if (withinTopHalf)
            topRightChild.containedParticlesIndices.push_back(containedParticlesIndices[i]);
        else if (withinLeftHalf)
            bottomLeftChild.containedParticlesIndices.push_back(containedParticlesIndices[i]);
        else
            bottomRightChild.containedParticlesIndices.push_back(containedParticlesIndices[i]);
    }

    // Top left child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (topLeftChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        topLeftChild.bucketIndex = bucketCounter++;
    topLeftChild.index = quadList.size();
    childrenIndices[0] = topLeftChild.index;
    quadList.push_back(topLeftChild);
    if (topLeftChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        quadList[topLeftChild.index].divide(quadList, pSystem, bucketCounter);

    // Top right child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (topRightChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        topRightChild.bucketIndex = bucketCounter++;
    topRightChild.index = quadList.size();
    childrenIndices[1] = topRightChild.index;
    quadList.push_back(topRightChild);
    if (topRightChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        quadList[topRightChild.index].divide(quadList, pSystem, bucketCounter);

    // Bottom left child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (bottomLeftChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        bottomLeftChild.bucketIndex = bucketCounter++;
    bottomLeftChild.index = quadList.size();
    childrenIndices[2] = bottomLeftChild.index;
    quadList.push_back(bottomLeftChild);
    if (bottomLeftChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        quadList[bottomLeftChild.index].divide(quadList, pSystem, bucketCounter);

    // Bottom right child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (bottomRightChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        bottomRightChild.bucketIndex = bucketCounter++;
    bottomRightChild.index = quadList.size();
    childrenIndices[3] = bottomRightChild.index;
    quadList.push_back(bottomRightChild);
    if (bottomRightChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        quadList[bottomRightChild.index].divide(quadList, pSystem, bucketCounter);

    // Since this quad has been divided, it can no longer be a bucket
    isBucket = false;
}

// Traverses quads depth-first to check if they are within H_CELL. If they are, store them in neibBucketIndices
void quad::neibSearchTraversal(std::vector<quad>& quadList, int currentIndex)
{
    quad currentQuad = quadList[currentIndex];
    if (quadRect.withinDistance(currentQuad.quadRect, hCell))
    {
        if (currentQuad.isBucket)
            neibBucketIndices.push_back(currentQuad.bucketIndex);
        else
        {
            for (int i = 0; i < 4; i++)
                neibSearchTraversal(quadList, currentQuad.childrenIndices[i]);
        }
    }
}
