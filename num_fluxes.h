/**
 * @file num_fluxes.h
 * @brief Contains function decls for normal numerical fluxes
 */

#include <deal.II/base/point.h>

#include "common.h"
#include "wind.h"

#ifndef num_fluxes_h
#define num_fluxes_h

Tensor<1,2> exact_flux(const double s_value, const Point<2> &loc);
double rusanov_flux(const double o_state, const double n_state, 
                         const Point<2> &loc, const Tensor<1,2> &normal);

#endif