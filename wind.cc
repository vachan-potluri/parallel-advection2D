/**
 * @file wind.cc
 * @brief Contains function defn for the advection velocity
 */

#include "wind.h"

/**
 * @brief Returns wind velocity vector taking a point as argument
 */
const Tensor<1,2> wind(const Point<2> &p)
{
        Tensor<1,2> wind_value;
        wind_value[0] = 1.0;
        wind_value[1] = 1.0;
        wind_value /= wind_value.norm();
        return wind_value;
}