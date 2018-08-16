#include <GL/glut.h>
#include "particleSystem.cuh"
#include "quadtreeSystem.cuh"
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
    glutCreateWindow("CUDA Quadree KNN");
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

// TODO: Don't forget quad neib finding is broken
//void displayHandler::drawNeibQuads(quadtreeSystem& QuadSystem,)

// Draws the Z-Order curve to represent the organization of bucket data to go the GPU
void displayHandler::drawMortonCurve(quadtreeSystem& QuadSystem, particleSystem& pSystem)
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
        // Draw morton-curve lines from Quad to Quad
        for (int i = 0; i < QuadSystem.quadList.size() - 1; i++)
        {
            // Colour related to location in memory
            glColor3f(0.4f * (float)i / (float)QuadSystem.quadList.size(), 0.2f, 0.2f);
            glVertex2f(QuadSystem.quadList[i].quadRect.centre.x, QuadSystem.quadList[i].quadRect.centre.y);
            glVertex2f(QuadSystem.quadList[i + 1].quadRect.centre.x, QuadSystem.quadList[i + 1].quadRect.centre.y);
        }
        glEnd();
    }
}

void displayHandler::drawNeibRadius(quadtreeSystem& QuadSystem, particleSystem& pSystem, int QuadIdx)
{
    glBegin(GL_LINES);
    glColor3f(0.3, 0.8f, 0.3f);
    float hCell = 2 * std::max(QuadSystem.quadList[QuadIdx].quadRect.width, QuadSystem.quadList[QuadIdx].quadRect.height);

    // Top range
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.centre.x, QuadSystem.quadList[QuadIdx].quadRect.topLeft.y);
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.centre.x, QuadSystem.quadList[QuadIdx].quadRect.topLeft.y + hCell);

    // Bottom range
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.centre.x, QuadSystem.quadList[QuadIdx].quadRect.bottomLeft.y);
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.centre.x, QuadSystem.quadList[QuadIdx].quadRect.bottomLeft.y - hCell);

    // Right range
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.topRight.x, QuadSystem.quadList[QuadIdx].quadRect.centre.y);
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.topRight.x + hCell, QuadSystem.quadList[QuadIdx].quadRect.centre.y);

    // Left range
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.topLeft.x, QuadSystem.quadList[QuadIdx].quadRect.centre.y);
    glVertex2f(QuadSystem.quadList[QuadIdx].quadRect.topLeft.x - hCell, QuadSystem.quadList[QuadIdx].quadRect.centre.y);

    glEnd();
}

// Traces lines clockwise to draw an quad
void displayHandler::drawQuad(quadtreeSystem& QuadSystem, int index)
{
    if (DRAW_BUCKETS)
    {
        glBegin(GL_LINES);
        glColor3f(
            (0.5f * (float)index / (float)QuadSystem.quadList.size()) / 2.0f,
            (0.2f) / 2.0f,
            (0.2f) / 2.0f
        );

        // Top side
        glVertex2f(QuadSystem.quadList[index].quadRect.topLeft.x, QuadSystem.quadList[index].quadRect.topLeft.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.topRight.x, QuadSystem.quadList[index].quadRect.topRight.y);
        // Right side
        glVertex2f(QuadSystem.quadList[index].quadRect.topRight.x, QuadSystem.quadList[index].quadRect.topRight.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.bottomRight.x, QuadSystem.quadList[index].quadRect.bottomRight.y);
        // Bottom side
        glVertex2f(QuadSystem.quadList[index].quadRect.bottomRight.x, QuadSystem.quadList[index].quadRect.bottomRight.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.bottomLeft.x, QuadSystem.quadList[index].quadRect.bottomLeft.y);
        // Left side
        glVertex2f(QuadSystem.quadList[index].quadRect.bottomLeft.x, QuadSystem.quadList[index].quadRect.bottomLeft.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.topLeft.x, QuadSystem.quadList[index].quadRect.topLeft.y);

        glEnd();
    }
}

void displayHandler::fillQuad(quadtreeSystem& QuadSystem, int index)
{
    if (DRAW_BUCKETS)
    {
        glBegin(GL_POLYGON);
        glColor3f(0.5f, 0.2f, 0.2f);
        glVertex2f(QuadSystem.quadList[index].quadRect.topLeft.x, QuadSystem.quadList[index].quadRect.topLeft.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.topRight.x, QuadSystem.quadList[index].quadRect.topRight.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.bottomRight.x, QuadSystem.quadList[index].quadRect.bottomRight.y);
        glVertex2f(QuadSystem.quadList[index].quadRect.bottomLeft.x, QuadSystem.quadList[index].quadRect.bottomLeft.y);
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
