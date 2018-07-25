#include <GL/glut.h>
#include "particleSystem.cuh"
#include "octreeSystem.cuh"
#include "simulationSettings.h"
#include "displayHandler.cuh"

void displayHandler::init()
{
    //int argc;
    //char* argv[];
    char *myargv [1];
    int myargc=1;
    myargv [0]=strdup(" ");
    // I have no idea how openGL works. This code is from https://stackoverflow.com/questions/42405420/how-to-draw-a-single-pixel-in-opengl
    glutInit(&myargc, myargv);
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("CUDA Octree KNN");
    glClearColor(10.0f / 255.0f, 10.0f / 255.0f, 10.0f / 255.0f, 0.0f);
    glColor3f(1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1000.0, 0.0, 1000.0);
    glMatrixMode(GL_MODELVIEW);
}

void displayHandler::drawCircle(float2 centre, float radius)
{
    glBegin(GL_LINE_LOOP);
    glColor3f(0.3f, 0.3f, 0.3f);
    // The amount of points drawn must scale with the radius so it looks smooth
    float points = 5 * radius;

    for (float i = 0; i < points; i++)
    {
        float angle = 2.0f * 3.14159f * i / points;
        float x = radius * cosf(angle);
        float y = radius * sinf(angle);
        glVertex2f(x + centre.x, y + centre.y);
    }

    glEnd();
}

// TODO: Don't forget octant neib finding is broken
//void displayHandler::drawNeibOcts(octreeSystem& octSystem,)

// Draws the Z-Order curve to represent the organization of bucket data to go the GPU
void displayHandler::drawMortonCurve(octreeSystem& octSystem)
{
    if (DRAW_MORTON_CURVE)
    {
        glBegin(GL_LINES);
        for (int i = 0; i < octSystem.octantList.size(); i++)
        {
            if (DRAW_MORTON_CURVE && i > 0)
            {
                glColor3f(
                    (0.5f * (float)i / (float)octSystem.octantList.size()) / 2.0f,
                    (0.2f) / 2.0f,
                    (0.2f) / 2.0f
                );
                glVertex2f(octSystem.octantList[i - 1].octRect.centre.x, octSystem.octantList[i - 1].octRect.centre.y);
                glVertex2f(octSystem.octantList[i].octRect.centre.x, octSystem.octantList[i].octRect.centre.y);
            }
        }
        glEnd();
    }
}

void displayHandler::drawNeibRadius(octreeSystem& octSystem, particleSystem& pSystem, int octIdx)
{
    glBegin(GL_LINES);
    glColor3f(0.3, 0.8f, 0.3f);
    float hCell = 2 * std::max(octSystem.octantList[octIdx].octRect.width, octSystem.octantList[octIdx].octRect.height);

    // Top range
    glVertex2f(octSystem.octantList[octIdx].octRect.centre.x, octSystem.octantList[octIdx].octRect.topLeft.y);
    glVertex2f(octSystem.octantList[octIdx].octRect.centre.x, octSystem.octantList[octIdx].octRect.topLeft.y + hCell);

    // Bottom range
    glVertex2f(octSystem.octantList[octIdx].octRect.centre.x, octSystem.octantList[octIdx].octRect.bottomLeft.y);
    glVertex2f(octSystem.octantList[octIdx].octRect.centre.x, octSystem.octantList[octIdx].octRect.bottomLeft.y - hCell);

    // Right range
    glVertex2f(octSystem.octantList[octIdx].octRect.topRight.x, octSystem.octantList[octIdx].octRect.centre.y);
    glVertex2f(octSystem.octantList[octIdx].octRect.topRight.x + hCell, octSystem.octantList[octIdx].octRect.centre.y);

    // Left range
    glVertex2f(octSystem.octantList[octIdx].octRect.topLeft.x, octSystem.octantList[octIdx].octRect.centre.y);
    glVertex2f(octSystem.octantList[octIdx].octRect.topLeft.x - hCell, octSystem.octantList[octIdx].octRect.centre.y);

    glEnd();
}

// Traces lines clockwise to draw an octant
void displayHandler::drawOct(octreeSystem& octSystem, int index)
{
    if (DRAW_BUCKETS)
    {
        glBegin(GL_LINES);
        glColor3f(
            (0.5f * (float)index / (float)octSystem.octantList.size()) / 2.0f,
            (0.2f) / 2.0f,
            (0.2f) / 2.0f
        );

        // Top side
        glVertex2f(octSystem.octantList[index].octRect.topLeft.x, octSystem.octantList[index].octRect.topLeft.y);
        glVertex2f(octSystem.octantList[index].octRect.topRight.x, octSystem.octantList[index].octRect.topRight.y);
        // Right side
        glVertex2f(octSystem.octantList[index].octRect.topRight.x, octSystem.octantList[index].octRect.topRight.y);
        glVertex2f(octSystem.octantList[index].octRect.bottomRight.x, octSystem.octantList[index].octRect.bottomRight.y);
        // Bottom side
        glVertex2f(octSystem.octantList[index].octRect.bottomRight.x, octSystem.octantList[index].octRect.bottomRight.y);
        glVertex2f(octSystem.octantList[index].octRect.bottomLeft.x, octSystem.octantList[index].octRect.bottomLeft.y);
        // Left side
        glVertex2f(octSystem.octantList[index].octRect.bottomLeft.x, octSystem.octantList[index].octRect.bottomLeft.y);
        glVertex2f(octSystem.octantList[index].octRect.topLeft.x, octSystem.octantList[index].octRect.topLeft.y);

        glEnd();
    }
}

void displayHandler::fillOct(octreeSystem& octSystem, int index)
{
    if (DRAW_BUCKETS)
    {
        glBegin(GL_POLYGON);
        glColor3f(0.5f, 0.2f, 0.2f);
        glVertex2f(octSystem.octantList[index].octRect.topLeft.x, octSystem.octantList[index].octRect.topLeft.y);
        glVertex2f(octSystem.octantList[index].octRect.topRight.x, octSystem.octantList[index].octRect.topRight.y);
        glVertex2f(octSystem.octantList[index].octRect.bottomRight.x, octSystem.octantList[index].octRect.bottomRight.y);
        glVertex2f(octSystem.octantList[index].octRect.bottomLeft.x, octSystem.octantList[index].octRect.bottomLeft.y);
        glEnd();
    }
}

// Fills a pixel at each particle's position
void displayHandler::drawParticle(particleSystem& pSystem, int i)
{
    if (DRAW_PARTICLES)
    {
        glBegin(GL_POINTS);
        typedef GLfloat point2[2];
        glColor3f(pSystem.densities[i] * 10, 0.5f, 0.5f);
        point2 particleDrawPoint = {pSystem.pos[i].x, pSystem.pos[i].y};
        glVertex2fv(particleDrawPoint);
        glEnd();
    }
}