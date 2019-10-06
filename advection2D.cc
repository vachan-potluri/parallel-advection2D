/**
 * @file advection2D.cc
 * @brief Defines advection2D class
 */

#include "advection2D.h"

/**
 * @brief Constructor with @p order of polynomial approx as arg
 * 
 * advection2D::mpi_communicator is initialised.
 * advection2D::mapping, advection2D::fe and advection2D::fe_face are initialised.
 * advection2D::dof_handler is associated to advection2D::triang.
 * advection2D::pcout is initialised to print with process rank 0
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
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator)==0))
{}

/**
 * @brief Sets up the system
 * 
 * 1. Mesh is setup and stored in advection2D::triang. Partition is done internally
 * 2. advection2D::dof_handler is linked to advection2D::fe
 * 3. advection2D::g_solution and advection2D::g_rhs are set using locally owned and relevant dofs
 */
void advection2D::setup_system()
{
        pcout << "Setting up the system" << std::endl;
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // partition happens automatically

        dof_handler.distribute_dofs(fe);
        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
        DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_locations);

        g_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        gold_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        g_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
}

/**
 * @brief Assembles the system
 * 
 * Calculating mass and differentiation matrices is as usual. Each face will have its own flux
 * matrix. The containers advection2D::face_first_dof and advection2D::face_dof_increment are used
 * to map face-local dof index to cell dof index.
 */
void advection2D::assemble_system()
{
        pcout << "Assembling system" << std::endl;
        // allocate all local matrices
        FullMatrix<double> l_mass(fe.dofs_per_cell),
                l_mass_inv(fe.dofs_per_cell),
                l_diff(fe.dofs_per_cell),
                l_flux(fe.dofs_per_cell),
                temp(fe.dofs_per_cell); // initialise with square matrix size
        QGauss<2> cell_quad_formula(fe.degree+1); // (N+1) gauss quad for cell
        QGauss<1> face_quad_formula(fe.degree+1); // for face
        FEValues<2> fe_values(fe, cell_quad_formula,
                update_values | update_gradients | update_JxW_values | update_quadrature_points);
        FEFaceValues<2> fe_face_values(fe, face_quad_formula,
                update_values | update_JxW_values | update_quadrature_points);
        
        uint i, j, i_face, j_face, qid, face_id;

        for(auto &cell: dof_handler.active_cell_iterators()){
                if(!cell->is_locally_owned()) continue; // skip if cell is not owned by this mpi proc

                std::cout << "Processor " << Utilities::MPI::this_mpi_process(mpi_communicator) <<
                " Cell " << cell->index() << std::endl;
                // stiffness matrix
                fe_values.reinit(cell);
                l_mass = 0;
                l_diff = 0;
                for(qid=0; qid<fe_values.n_quadrature_points; qid++){
                        for(i=0; i<fe.dofs_per_cell; i++){
                                for(j=0; j<fe.dofs_per_cell; j++){
                                        l_mass(i,j) += fe_values.shape_value(i, qid) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                        l_diff(i,j) += fe_values.shape_grad(i, qid) *
                                                wind(fe_values.quadrature_point(qid)) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                } // inner loop cell shape fns
                        } // outer loop cell shape fns
                } // loop over cell quad points
                l_mass_inv.invert(l_mass);
                l_mass_inv.mmult(temp, l_diff); // store mass_inv * diff into temp
                stiff_mats[cell->index()] = temp;

                // lifting matrices of 4 faces
                // each face will have a separate flux matrix
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        fe_face_values.reinit(cell, face_id);
                        l_flux = 0;
                        for(qid=0; qid<fe_face_values.n_quadrature_points; qid++){
                                for(i_face=0; i_face<fe_face.dofs_per_face; i_face++){
                                        for(j_face=0; j_face<fe_face.dofs_per_face; j_face++){
                                                // mapping
                                                i = face_first_dof[face_id] +
                                                        i_face*face_dof_increment[face_id];
                                                j = face_first_dof[face_id] +
                                                        j_face*face_dof_increment[face_id];
                                                l_flux(i,j) +=
                                                        fe_face_values.shape_value(i, qid) *
                                                        fe_face_values.shape_value(j, qid) *
                                                        fe_face_values.JxW(qid);
                                        } // inner loop over face shape fns
                                } // outer loop over face shape fns
                        } // loop over face quad points
                        l_mass_inv.mmult(temp, l_flux);
                        lift_mats[cell->index()][face_id] = temp;
                }// loop over faces
        } // loop over locally owned cells
}



// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
// Test function
// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#ifdef DEBUG
void advection2D::test()
{
        // deallog << "---------------------------------------------" << std::endl;
        // deallog << "Testing advection2D class" << std::endl;
        // deallog << "---------------------------------------------" << std::endl;
        advection2D problem(1);
        problem.setup_system();
        MPI_Barrier(MPI_COMM_WORLD);
        problem.assemble_system();
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
