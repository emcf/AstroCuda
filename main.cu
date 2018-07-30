// This is an octree-based GPU SPH demo
// Execute the build.sh script to compile and run the code.
// Written by Emmett McFarlane

#include "octreeSystem.cuh"
#include "particleSystem.cuh"
#include "physicsSystemGPU.cuh"
#include "displayHandler.cuh"
#include <GL/glut.h>

static octreeSystem octSystem;
static particleSystem pSystem;
static physicsSystemGPU sphSystem;
static displayHandler display;

void draw()
{
    glClear(GL_COLOR_BUFFER_BIT);
    display.drawMortonCurve(octSystem);
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
        pSystem.integrate();
        octSystem.reset();
        octSystem.makeOctree(pSystem);
        octSystem.findAllBucketNeibs();
        octSystem.eliminateBranches();

        deviceOctant* d_octantList = octSystem.sendToGPU();
        sphSystem.computeDensity(octSystem, pSystem, d_octantList);
        octSystem.freeFromGPU();
        cudaMemcpy(pSystem.densities, pSystem.d_densities, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pSystem.smoothingLengths, pSystem.d_smoothingLengths, N*sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; i++)
            printf("%f %f\n", pSystem.smoothingLengths[i], pSystem.densities[i]);

        draw();
    }
}
