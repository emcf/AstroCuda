// This is an quadtree-based GPU SPH demo
// Execute the build.sh script to compile and run the code.
// Written by Emmett McFarlane

#include "quadtreeSystem.cuh"
#include "particleSystem.cuh"
#include "physicsSystemGPU.cuh"
#include "displayHandler.cuh"
#include <GL/glut.h>
#include <time.h>

static quadtreeSystem quadSystem;
static particleSystem pSystem;
static physicsSystemGPU sphSystem;
static displayHandler display;

void draw()
{
    glClear(GL_COLOR_BUFFER_BIT);
    display.drawMortonCurve(quadSystem, pSystem);
    display.drawSmoothingLenghs(pSystem);
    for (int i = 0; i < quadSystem.quadCount; i++)
        display.drawQuad(quadSystem, i);
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
        // Build quadtree on host
        quadSystem.reset();
        quadSystem.makeQuadtree(pSystem);
        quadSystem.findAllBucketNeibs();
        quadSystem.eliminateBranches();

        // Allocate, rearrange, and send particle data to GPU
        deviceParticle* d_deviceParticleList;
        deviceQuad* d_quadList;
        cudaMalloc((void**) &d_deviceParticleList, N*sizeof(deviceParticle));
        cudaMalloc((void**) &d_quadList, quadSystem.quadCount*sizeof(deviceQuad));
        quadSystem.arrangeAndTransfer(pSystem, d_quadList, d_deviceParticleList);

        // Compute density and integrate positions on GPU
        sphSystem.solveSPH(quadSystem, d_quadList, d_deviceParticleList);
        sphSystem.integrate(quadSystem, d_quadList, d_deviceParticleList);

        // Return data from GPU, rearrange particle data to work on host
        quadSystem.getFromGPU(d_quadList);
        pSystem.getFromGPU(d_deviceParticleList);

        draw();
    }
}
