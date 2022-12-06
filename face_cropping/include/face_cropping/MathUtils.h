#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>

namespace face_cropping
{
    int getClosestNumberDividableBy(float a, float b)
    {
        float n1 = a;
        while (fabsf(roundf(n1 / b) - n1 / b) > 0.00001f)
        {
            n1 += 1;
        }
        float n2 = a;
        while (fabsf(roundf(n2 / b) - n2 / b) > 0.00001f)
        {
            n2 -= 1;
        }
        if (abs(n1 - a) > abs(a - n2))
        {
            return n1;
        }
        return n2;
    }
}
#endif
