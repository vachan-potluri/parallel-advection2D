/**
 * @file advection2D.cc
 * @brief Defines advection2D class
 */

#include "advection2D.h"

/**
 * @brief Constructor with @p order of polynomial approx as arg
 * 
 * advection2D::mapping, advection2D::fe and advection2D::fe_face are initialised.
 * advection2D::dof_handler is associated to advection2D::triang.
 * Based on order, face_first_dof and face_dof_increment containers are set here. See
 * https://www.dealii.org/current/doxygen/deal.II/structGeometryInfo.html and
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html for face and dof ordering
 * respectively in a cell. According to GeometryInfo, the direction of face lines is along the
 * positive axes. See DG notes dated 24-09-19.
 * 
 * Eg: for order=2, on 1-th face, the first cell dof is 2 and the next dof is obtained after
 * increment of 3
 */
advection2D::advection2D(const uint order)
: mpi_communicator(MPI_COMM_WORLD),
        mapping(), fe(order), fe_face(order),
        triang(mpi_communicator), dof_handler(triang),
        face_first_dof{0, order, 0, (order+1)*order},
        face_dof_increment{order+1, order+1, 1, 1},
        pcout(std::cout, (Utilities::MPI::this_mpi_process==0))
{}



// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
// Test function
// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#ifdef DEBUG
void advection2D::test()
{
        // deallog << "---------------------------------------------" << std::endl;
        // deallog << "Testing advection2D class" << std::endl;
        // deallog << "---------------------------------------------" << std::endl;
        // advection2D problem(1);
        // problem.setup_system();
        // problem.assemble_system();
        // problem.print_matrices();
        // problem.set_IC();
        // problem.set_boundary_ids();

        // double start_time = 0.0, end_time = 0.5, time_step = 0.005;
        // uint time_counter = 0;
        // std::string base_filename = "output.vtk";
        // problem.output(base_filename + ".0"); // initial condition
        // for(double cur_time = start_time; cur_time<end_time; cur_time+=time_step){
        //         deallog << "Step " << time_counter << " time " << cur_time << std::endl;
        //         problem.update(time_step);
        //         time_counter++;
        //         problem.output(base_filename + "." + std::to_string(time_counter));
        // }
}
#endif
