#pragma once
#include <algorithm>

struct quadRectangle
{
    // Constructor/Destructor
    quadRectangle();
    ~quadRectangle();

    // Geometric data
    float width;
    float height;
    float2 topLeft;
    float2 topRight;
    float2 bottomRight;
    float2 bottomLeft;
    float2 centre;

    float left();
    float right();
    float top();
    float bottom();
    // Returns true if the distance between the closest sides of this rect and rectToCompare is less than maxDistance
    bool withinDistance(quadRectangle rectToCompare, float maxDistance);
    // Returns true if thie rectToCompare is inside this rect
    bool contains(quadRectangle rectToCompare);
    void calculateVertices();
};
