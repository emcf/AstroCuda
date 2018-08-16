// This is an octree-based GPU SPH demo
// Execute the build.sh script to compile and run the code.
// Written by Emmett McFarlane

#include "octreeSystem.cuh"
#include "particleSystem.cuh"
#include "physicsSystemGPU.cuh"
#include "displayHandler.cuh"
#include <GL/glut.h>
#include <time.h>

static octreeSystem octSystem;
static particleSystem pSystem;
static physicsSystemGPU sphSystem;
static displayHandler display;

void draw()
{
    glClear(GL_COLOR_BUFFER_BIT);
    display.drawMortonCurve(octSystem, pSystem);
    display.drawSmoothingLenghs(pSystem);
    for (int i = 0; i < octSystem.octCount; i++)
        display.drawOct(octSystem, i);
    for (int i = 0; i < N; i++)
        display.drawParticle(pSystem, i);
    glFlush();
}

int main(int argc, char* argv[])
{
    pSystem.init();
    display.init();
    glutDisplayFunc(draw);

    while (true)
    {
        // Build octree on host
        octSystem.reset();
        octSystem.makeOctree(pSystem);
        octSystem.findAllBucketNeibs();
        octSystem.eliminateBranches();

        // Allocate, rearrange, and send particle data to GPU
        deviceParticle* d_deviceParticleList;
        deviceOctant* d_octantList;
        cudaMalloc((void**) &d_deviceParticleList, N*sizeof(deviceParticle));
        cudaMalloc((void**) &d_octantList, octSystem.octCount*sizeof(deviceOctant));
        octSystem.arrangeAndTransfer(pSystem, d_octantList, d_deviceParticleList);

        // Compute density and integrate positions on GPU
        sphSystem.solveSPH(octSystem, d_octantList, d_deviceParticleList);
        sphSystem.integrate(octSystem, d_octantList, d_deviceParticleList);

        // Return data from GPU, rearrange particle data to work on host
        octSystem.getFromGPU(d_octantList);
        pSystem.getFromGPU(d_deviceParticleList);

        draw();
    }
}
