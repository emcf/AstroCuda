#pragma once
#include <GL/glut.h>
#include "simulationSettings.h"
#include "particleSystem.cuh"
#include "quadtreeSystem.cuh"

struct displayHandler
{
    void init();
    void drawSimulation();
    void drawMortonCurve(quadtreeSystem& quadSystem, particleSystem& pSystem);
    void drawCircle(float2 centre, float radius);
    void drawQuad(quadtreeSystem& quadSystem, int index);
    void fillQuad(quadtreeSystem& quadSystem, int index);
    void drawSmoothingLenghs(particleSystem& pSystem);
    void drawParticle(particleSystem& pSystem, int index);
    void fillCircle(GLfloat x, GLfloat y, GLfloat radius);
    void drawNeibRadius(quadtreeSystem& quadSystem, particleSystem& pSystem, int bucketIndex);
};
