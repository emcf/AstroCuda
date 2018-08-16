#include <GL/glut.h>
#include "particleSystem.cuh"
#include "octreeSystem.cuh"
#include "simulationSettings.h"
#include "displayHandler.cuh"
#define PI 3.14159f

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
    //glColor3f(0.3f, 0.3f, 0.3f);
    glColor3f(40.0f / 255.0f, 40.0f / 255.0f, 40.0f / 255.0f);
    // The amount of points drawn must scale with the radius so it looks smooth. Max is 100 to minimize lag
    float points = PI * radius;
    points = (points > 50) ? 50 : points;

    for (float i = 0; i < points; i++)
    {
        float angle = 2.0f * PI * i / points;
        float x = radius * cosf(angle);
        float y = radius * sinf(angle);
        glVertex2f(x + centre.x, y + centre.y);
    }

    glEnd();
}

// TODO: Don't forget octant neib finding is broken
//void displayHandler::drawNeibOcts(octreeSystem& octSystem,)

// Draws the Z-Order curve to represent the organization of bucket data to go the GPU
void displayHandler::drawMortonCurve(octreeSystem& octSystem, particleSystem& pSystem)
{
    if (DRAW_MORTON_CURVE)
    {
        glBegin(GL_LINES);
        // Draw morton-curve lines from particle to particle
        for (int i = 0; i < N - 1; i++)
        {
            // Colour related to location in memory
            glColor3f(0.2f * (float)i / (float)N, 0.1f, 0.1f);
            float2 pos = {pSystem.h_deviceParticleList[i].particleData[0], pSystem.h_deviceParticleList[i].particleData[1]};
            float2 pos2 = {pSystem.h_deviceParticleList[i + 1].particleData[0], pSystem.h_deviceParticleList[i +1 ].particleData[1]};
            glVertex2f(pos.x, pos.y);
            glVertex2f(pos2.x, pos2.y);
        }
        // Draw morton-curve lines from oct to oct
        for (int i = 0; i < octSystem.octantList.size() - 1; i++)
        {
            // Colour related to location in memory
            glColor3f(0.4f * (float)i / (float)octSystem.octantList.size(), 0.2f, 0.2f);
            glVertex2f(octSystem.octantList[i].octRect.centre.x, octSystem.octantList[i].octRect.centre.y);
            glVertex2f(octSystem.octantList[i + 1].octRect.centre.x, octSystem.octantList[i + 1].octRect.centre.y);
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

void displayHandler::drawSmoothingLenghs(particleSystem& pSystem)
{
    if (DRAW_SMOOTHING)
    {
        for (int i = 0; i < N; i++)
        {
            float2 pos;
            pos.x = pSystem.pos[i].x;
            pos.y = pSystem.pos[i].y;
            drawCircle(pos, pSystem.smoothingLengths[i]);
        }
    }
}

// Fills a pixel at each particle's position
void displayHandler::drawParticle(particleSystem& pSystem, int i)
{
    glColor3f(pSystem.densities[i] * 5, 0.2f, 0.2f);
    fillCircle(pSystem.pos[i].x, pSystem.pos[i].y, pSystem.mass[i] * 0.5f);
}

void displayHandler::fillCircle(GLfloat x, GLfloat y, GLfloat radius)
{
	int triangles = 10;
	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(x, y);
	for (int i = 0; i <= triangles; i++)
		glVertex2f(x + (radius * cos(i * 2 * PI / triangles)), y + (radius * sin(i * 2 * PI / triangles)));
	glEnd();
}
