/**
 * @file advection2D.h
 * @brief Defines advection2D class
 */

// Includes: most of them are from step-12 and dflo
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>

#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <functional>

// #include <deal.II/numerics/derivative_approximation.h> // for adaptive mesh

#include "common.h"
#include "wind.h"
#include "IC.h"
#include "BCs.h"
#include "num_fluxes.h"

#ifndef advection2D_h
#define advection2D_h

/**
 * @class advection2D
 * @brief A class for 2D linear advection equation
 * 
 * The problem to be solved is
 * @f[ \frac{\partial \phi}{\partial t} + \nabla \cdot (\phi \vec{v}) = 0 @f]
 * @f$ \phi @f$ is the variable and @f$\vec{v}@f$ is the advecting wind. The DG formulation of this
 * problem is
 * @f[
 * \int_{\Omega_h} l_j \left(\sum \frac{\partial\phi_i}{\partial t} l_i\right) \,d\Omega +
 * \sum_{\text{faces}} \int_{\text{face}} l_j \left(\sum\phi^*_i l_i\right)
 * \vec{v}\cdot\vec{n}\,dA -
 * \int_{\Omega_h}\nabla l_j\cdot\vec{v} \left(\sum\phi_i l_i\right) \,d\Omega = 0
 * @f]
 * Explicit time integration gives
 * @f[
 * [M]\{\phi\}^{n+1} = [M]\{\phi\}^n + \left( [D]\{\phi\}^n - \sum_{\text{faces}}[F]\{f^*\}^n \right)
 * \Delta t
 * @f]
 * @f$[M]@f$ is the mass matrix, @f$[D]@f$ is the differentiation matrix and @f$[F]@f$ is the flux
 * matrix. Multiplying by mass inverse:
 * @f[
 * \{\phi\}^{n+1} = \{\phi\}^n + \left( [S]\{\phi\}^n - \sum_{\text{faces}}[L]\{f^*\}^n \right)
 * @f]
 * Here @f$[S]@f$ is the stiffness matrix and @f$[L]@f$ is the lifting matrix
 * @note Every face will have its own lifting matrix. The contribution of cell vertices from two
 * different faces cannot be clubbed into a single lifting matrix because two numerical fluxes act
 * at every cell vertex. Accordingly, 4 different numerical flux vectors will multiply these 4
 * lifting matrices. See ME757 material "Notes14.pdf"
 * 
 * @remark The code is working fine, but the problem considered requires a limiter.
 * 
 * @todo Add function for calculating stable time step
 * @todo Add limiter functionality
 */

class advection2D
{

        public:
        advection2D(const uint order);
        // first cell dof on a face
        const std::array<uint, GeometryInfo<2>::faces_per_cell> face_first_dof;
        // increment of cell dof on a face
        const std::array<uint, GeometryInfo<2>::faces_per_cell> face_dof_increment;
        std::array< std::function<double(const double)>, 3 > bc_fns = {b0,b1,b2};

        private:
        void setup_system();
        void assemble_system();
        void set_IC();
        void set_boundary_ids();
        void update(const double time_step);
        void print_matrices() const;
        void output(const std::string &filename) const;

        // class variables
        Triangulation<2> triang;
        const MappingQ1<2> mapping;

        // By default, fe assumes all dofs to be inside cell. Thus, fe.dofs_per_face will return 0.
        // The variable fe_face can be thought as projection of a DG basis on a face
        FE_DGQ<2> fe;
        FE_FaceQ<2> fe_face; // face finite element
        DoFHandler<2> dof_handler;
        std::vector<Point<2>> dof_locations; // all dof locations

        // solution has to be global to enable results output, a local solution cannot used to
        // output results
        Vector<double> g_solution; // global solution
        Vector<double> gold_solution; // global old solution
        std::vector<Vector<double>> l_rhs; // local rhs of every cell

        // stiffness and lifting matrices
        std::vector<FullMatrix<double>> stiff_mats;
        std::vector< std::array<FullMatrix<double>, GeometryInfo<2>::faces_per_cell> > lift_mats;



        public:
        #ifdef DEBUG
        static void test();
        #endif
};

#endif
