#include "octreeSystem.cuh"
#include <stdio.h>

// Constructor
octreeSystem::octreeSystem()
{

}

// Destructor
octreeSystem::~octreeSystem()
{

}

void octreeSystem::reset()
{
    // initialize octreeSystem
    octantList.clear();
    bucketCounter = 0;

    // Initialize the parent oct's cell
    octRectangle parentRect;
    parentRect.topLeft.x = 0;
    parentRect.topLeft.y = HEIGHT;
    parentRect.height = HEIGHT;
    parentRect.width = WIDTH;
    parentRect.calculateVertices();

    // Initialize the parent oct
    octant parentOct;
    parentOct.index = 0;
    parentOct.octRect = parentRect;
    parentOct.isBucket = true;
    parentOct.containedParticlesIndices.clear();

    // Reserve enough space for a fixed number of octants. This is required since dynamically allocating the
    // vector from a void inside one of the vector's contained objects will change memory addresses to the contained object,
    // The problem is that C++ will still use the old memory address for obtaining variables stored by that object.
    octantList.reserve(OCT_LIST_ALLOC);
    octantList.push_back(parentOct);
}

void octreeSystem::makeOctree(particleSystem& pSystem)
{
    // set parent oct's contained particles to 1..N
    for (int i = 0; i < N; i++)
    {
        bool withinWidth = (octantList[0].octRect.topLeft.x < pSystem.pos[i].x && pSystem.pos[i].x < octantList[0].octRect.topRight.x);
        bool withinHeight = (octantList[0].octRect.bottomLeft.y < pSystem.pos[i].y && pSystem.pos[i].y < octantList[0].octRect.topLeft.y);
        if (withinWidth && withinHeight)
            octantList[0].containedParticlesIndices.push_back(i);
    }

    // Divide the first octant
    if (octantList[0].containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
    {
        octantList[0].isBucket = false;
        octantList[0].divide(octantList, pSystem, bucketCounter);
    }
}

// Arranges the particles in pSystem into a morton curve to send to GPU
void octreeSystem::arrangeAndTransfer(particleSystem& pSystem, deviceOctant* d_octantList, deviceParticle* d_deviceParticleList)
{
    pSystem.h_deviceParticleList = new deviceParticle[N];
    h_octantList = new deviceOctant[octCount];

    int MortonCounter = 0;
    for (int i = 0; i < octCount; i++)
    {
        int containedParticlesCount = octantList[i].containedParticlesIndices.size();

        // The neighbour search radius for this octant should be roughly proportional to this octant's size.
        // This is because extremely large octants are likely in less dense areas and require a larger search radius, and vice versa
        // Note: hCell should be an overestimate rather than an underestimate for accurate SPH
        float hCell = 2 * std::max(octantList[i].octRect.width, octantList[i].octRect.height) + 10;
        h_octantList[i].neibSearchRadius = hCell;

        // Transfers info storing the # of contained particles as well as # of neib buckets for this bucket
        h_octantList[i].containedParticleCount = containedParticlesCount;
        h_octantList[i].neibBucketCount = octantList[i].neibBucketIndices.size();

        // Allocate and transfer contained particle indices to the GPU
        h_octantList[i].firstContainedParticleIdx = MortonCounter;

        // Allocate and transfer neib bucket indices to the GPU
        cudaMalloc((void**) &(h_octantList[i].d_neibBucketsIndices), h_octantList[i].neibBucketCount * sizeof(int));
        cudaMemcpy(h_octantList[i].d_neibBucketsIndices, &(octantList[i].neibBucketIndices[0]), h_octantList[i].neibBucketCount * sizeof(int), cudaMemcpyHostToDevice);

        for (int j = 0; j < containedParticlesCount; j++)
        {
            // Convert particle info into congruent arrays
            deviceParticle h_particle;
            int particleIdx = octantList[i].containedParticlesIndices[j];
            h_particle.particleData[0] = pSystem.pos[particleIdx].x;
            h_particle.particleData[1] = pSystem.pos[particleIdx].y;
            h_particle.particleData[2] = pSystem.pos[particleIdx].z;
            h_particle.particleData[3] = particleIdx;
            h_particle.particleData[4] = pSystem.prevpos[particleIdx].x;
            h_particle.particleData[5] = pSystem.prevpos[particleIdx].y;
            h_particle.particleData[6] = pSystem.prevpos[particleIdx].z;
            h_particle.particleData[7] = pSystem.mass[particleIdx];
            h_particle.particleData[8] = pSystem.smoothingLengths[particleIdx];
            h_particle.particleData[9] = pSystem.densities[particleIdx];
            h_particle.particleData[10] = pSystem.omegas[particleIdx];
            h_particle.particleData[11] = pSystem.pressures[particleIdx];
            pSystem.h_deviceParticleList[MortonCounter] = h_particle;
            MortonCounter++;
        }
    }

    // OctantList data is on GPU, now transfer the "meta data" to the GPU
    cudaMemcpy(d_octantList, h_octantList, octCount*sizeof(deviceOctant), cudaMemcpyHostToDevice);
    // Transfer rearranged particle data to GPU
    cudaMemcpy(d_deviceParticleList, pSystem.h_deviceParticleList, N*sizeof(deviceParticle), cudaMemcpyHostToDevice);
}

// Converts each bucket octant into a deviceOctant and sends it to the device.
/*deviceOctant* octreeSystem::sendToGPU()
{
    // Allocate device octant on device and record their "meta data" (pointers) on the CPU
    h_octantList = new deviceOctant[octCount];
    deviceOctant* d_octantList;

    for (int i = 0; i < octCount; i++)
    {

    }

    // OctantList data is on GPU, now transfer the "meta data" to the GPU
    cudaMalloc((void**) &d_octantList, octCount*sizeof(deviceOctant));
    cudaMemcpy(d_octantList, h_octantList, octCount*sizeof(deviceOctant), cudaMemcpyHostToDevice);

    return d_octantList;
}*/

void octreeSystem::freeFromGPU()
{
    // Free dyanmically allocated memory
    for (int i = 0; i < octCount; i++)
    {
        cudaFree(h_octantList[i].d_neibBucketsIndices);
    }

    delete[] h_octantList;
}


void octreeSystem::eliminateBranches()
{
    for (int i = 0; i < octantList.size(); i+=0)
    {
        if (octantList[i].isBucket)
            i++;
        else
            octantList.erase(octantList.begin() + i);
    }

    octCount = octantList.size();
}

// Find neighbour buckets for all bucket octants. Complexity is O(N*log(N))
void octreeSystem::findAllBucketNeibs()
{
    for (int i = 0; i < octantList.size(); i++)
    {
        if (octantList[i].isBucket)
            octantList[i].neibSearchTraversal(octantList, 0);
    }
}

// Constructor
octant::octant()
{

}

// Destructor
octant::~octant()
{

}

void octant::divide(std::vector<octant>& octantList, particleSystem& pSystem, int& bucketCounter)
{
    // Top left
    octant topLeftChild;
    //topLeftChild.parentIndex = index;
    topLeftChild.octRect.topLeft = octRect.topLeft;
    topLeftChild.octRect.width = octRect.width / 2.00f;
    topLeftChild.octRect.height = octRect.height / 2.00f;
    topLeftChild.octRect.calculateVertices();
    // Top right
    octant topRightChild;
    //topRightChild.parentIndex = index;
    topRightChild.octRect.topLeft.x = octRect.centre.x;
    topRightChild.octRect.topLeft.y = octRect.topRight.y;
    topRightChild.octRect.width = octRect.width / 2.00f;
    topRightChild.octRect.height = octRect.height / 2.00f;
    topRightChild.octRect.calculateVertices();
    // Bottom left
    octant bottomLeftChild;
    //bottomLeftChild.parentIndex = index;
    bottomLeftChild.octRect.topLeft.x = octRect.topLeft.x;
    bottomLeftChild.octRect.topLeft.y =  octRect.centre.y;
    bottomLeftChild.octRect.width = octRect.width / 2.00f;
    bottomLeftChild.octRect.height = octRect.height / 2.00f;
    bottomLeftChild.octRect.calculateVertices();
    // Bottom right
    octant bottomRightChild;
    //bottomRightChild.parentIndex = index;
    bottomRightChild.octRect.topLeft = octRect.centre;
    bottomRightChild.octRect.width = octRect.width / 2.00f;
    bottomRightChild.octRect.height = octRect.height / 2.00f;
    bottomRightChild.octRect.calculateVertices();

    // Assign particles to each new child oct
    for (int i = 0; i < containedParticlesIndices.size(); i++)
    {
        bool withinTopHalf =  pSystem.pos[containedParticlesIndices[i]].y > octRect.centre.y;
        bool withinLeftHalf =  pSystem.pos[containedParticlesIndices[i]].x < octRect.centre.x;

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
    topLeftChild.index = octantList.size();
    childrenIndices[0] = topLeftChild.index;
    octantList.push_back(topLeftChild);
    if (topLeftChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        octantList[topLeftChild.index].divide(octantList, pSystem, bucketCounter);

    // Top right child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (topRightChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        topRightChild.bucketIndex = bucketCounter++;
    topRightChild.index = octantList.size();
    childrenIndices[1] = topRightChild.index;
    octantList.push_back(topRightChild);
    if (topRightChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        octantList[topRightChild.index].divide(octantList, pSystem, bucketCounter);

    // Bottom left child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (bottomLeftChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        bottomLeftChild.bucketIndex = bucketCounter++;
    bottomLeftChild.index = octantList.size();
    childrenIndices[2] = bottomLeftChild.index;
    octantList.push_back(bottomLeftChild);
    if (bottomLeftChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        octantList[bottomLeftChild.index].divide(octantList, pSystem, bucketCounter);

    // Bottom right child
    // If this child is a bucket, set its bucket index and assign this bucket index to its contained particles
    if (bottomRightChild.containedParticlesIndices.size() <= MAX_PARTICLES_PER_BUCKET)
        bottomRightChild.bucketIndex = bucketCounter++;
    bottomRightChild.index = octantList.size();
    childrenIndices[3] = bottomRightChild.index;
    octantList.push_back(bottomRightChild);
    if (bottomRightChild.containedParticlesIndices.size() > MAX_PARTICLES_PER_BUCKET)
        octantList[bottomRightChild.index].divide(octantList, pSystem, bucketCounter);

    // Since this oct has been divided, it can no longer be a bucket
    isBucket = false;
}

// Traverses octants recursively to check if they are within H_CELL. If they are, store them in neibBucketIndices
void octant::neibSearchTraversal(std::vector<octant>& octantList, int currentIndex)
{
    octant currentOct = octantList[currentIndex];
    float hCell = 2 * std::max(octRect.width, octRect.height) + 10;
    if (octRect.withinDistance(currentOct.octRect, hCell))
    {
        if (currentOct.isBucket)
            neibBucketIndices.push_back(currentOct.bucketIndex);
        else
        {
            for (int i = 0; i < 4; i++)
                neibSearchTraversal(octantList, currentOct.childrenIndices[i]);
        }
    }
}
