/**
 * @file wind.h
 * @brief Contains function decl for the advection velocity
 */

// #include <deal.II/base/tensor.h> // not reqd if point.h is included
#include <deal.II/base/point.h>

#include "common.h"

#ifndef wind_h
#define wind_h

const Tensor<1,2> wind(const Point<2> &p);

#endif