#include "octRectangle.cuh"

// Constructor
octRectangle::octRectangle()
{

}

// Destructor
octRectangle::~octRectangle()
{

}

float octRectangle::left() { return topLeft.x; }
float octRectangle::right() { return topRight.x; }
float octRectangle::top() { return topLeft.y; }
float octRectangle::bottom() { return bottomLeft.y; }

// Returns true if the distance between the closest sides of this rect and rectToCompare is less than maxDistance
bool octRectangle::withinDistance(octRectangle rectToCompare, float maxDistance)
{
    // In the below example, the closest side of rect A is the right side, and the closest side of rect B is the left side,
    // Thus these sides are used for the comparison.
    /*
      _____                  _____
     |     |       dx       |     |
     |  A  |<-------------->|  B  |
     |_____|                |_____|

    */

    // Find the distances between closest sides of each rectangle in each dimension
    float dxp = rectToCompare.left() - right();
    float dxm = left() - rectToCompare.right();
    float dx = (dxp > 0) ? dxp : (dxm > 0) ? dxm : 0;

    float dyp = rectToCompare.bottom() - top();
    float dym = bottom() - rectToCompare.top();
    float dy = (dyp > 0) ? dyp : (dym > 0) ? dym : 0;

    // Pythagorean theorem
    float squaredDistance = dx * dx + dy * dy;
    return squaredDistance <= (maxDistance * maxDistance);
}

// Returns true if thie rectToCompare is inside this rect
bool octRectangle::contains(octRectangle rectToCompare)
{
    return bottom() <= rectToCompare.bottom() && rectToCompare.top() <= top() && left() <= rectToCompare.left() && rectToCompare.right() <= right();
}

void octRectangle::calculateVertices()
{
    // Assume topLeft point and size are known
    topRight.x = topLeft.x + width;
    topRight.y = topLeft.y;

    bottomLeft.x = topLeft.x;
    bottomLeft.y = topLeft.y - height;

    bottomRight.x = topRight.x;
    bottomRight.y = bottomLeft.y;

    centre.x = (topLeft.x + bottomRight.x) / 2.0f;
    centre.y = (topLeft.y + bottomRight.y) / 2.0f;
}
