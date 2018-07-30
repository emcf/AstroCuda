#pragma once
#include <GL/glut.h>
#include "simulationSettings.h"
#include "particleSystem.cuh"
#include "octreeSystem.cuh"

struct displayHandler
{
    void init();
    void drawSimulation();
    void drawMortonCurve(octreeSystem& octSystem);
    void drawCircle(float2 centre, float radius);
    void drawOct(octreeSystem& octSystem, int index);
    void fillOct(octreeSystem& octSystem, int index);
    void drawSmoothingLenghs(particleSystem& pSystem);
    void drawParticle(particleSystem& pSystem, int index);
    void fillCircle(GLfloat x, GLfloat y, GLfloat radius);
    void drawNeibRadius(octreeSystem& octSystem, particleSystem& pSystem, int bucketIndex);
};
