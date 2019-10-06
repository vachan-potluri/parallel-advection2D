/**
 * @file num_fluxes.cc
 * @brief Contains function defns for normal numerical fluxes
 */

#include "num_fluxes.h"

/**
 * @brief Calculates exact flux based on the "s"tate value and "loc"ation
 */
Tensor<1,2> exact_flux(const double s_value, const Point<2> &loc)
{
        return s_value * wind(loc);
}

/**
 * @brief Calculates Rusanov numerical flux
 * @param[in] o_state The owner state
 * @param[in] n_state The neighbour state
 * @param[in] normal The face normal
 * @param[in] loc The quad point location
 * 
 * See https://www.cfd-online.com/Forums/blogs/praveen/315-flux-computation-unstructured-grids.html
 * and normal_numerical_flux function of step-33
 * @note The flux vector dotted with normal vector is returned. This is what is required for assembly
 * @warning The @p normal must be a unit vector and must point from owner to neighbor
 */
double rusanov_flux(const double o_state, const double n_state, 
                         const Point<2> &loc, const Tensor<1,2> &normal)
{
        const double num_visc = fabs(wind(loc)*normal); // artificial or numerical viscosity
        return 0.5*( exact_flux(o_state, loc) + exact_flux(n_state, loc) ) * normal +
               0.5*num_visc*(o_state - n_state);
}