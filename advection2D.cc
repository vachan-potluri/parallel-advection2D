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
: mapping(), fe(order), fe_face(order), dof_handler(triang),
        face_first_dof{0, order, 0, (order+1)*order},
        face_dof_increment{order+1, order+1, 1, 1}
{}

/**
 * @brief Sets up the system
 * 
 * 1. Mesh is setup and stored in advection2D::triang
 * 2. advection2D::dof_handler is linked to advection2D::fe
 * 3. advection2D::g_solution and advection2D::l_rhs sizes are set
 * 4. Sizes of advection2D::stiff_mats, advection2D::lift_mats and advection2D::l_rhs containers are
 * set
 */
void advection2D::setup_system()
{
        deallog << "Setting up the system" << std::endl;
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // 2^5=32 cells in each direction, total length 1m

        // set dof_handler
        dof_handler.distribute_dofs(fe);
        dof_locations.resize(dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler, dof_locations);

        // no system_matrix because the solution is updated cell wise
        g_solution.reinit(dof_handler.n_dofs());
        gold_solution.reinit(dof_handler.n_dofs());

        // set user flags for cell
        // for a face, cell with lower user index will be treated owner
        // is this reqd? can't we just use cell->index()?
        // uint i=0;
        // for(auto &cell: dof_handler.active_cell_iterators()){
        //         cell->set_user_index(i++);
        // } // loop over cells

        // set sizes of stiffness and lifting matrix containers
        stiff_mats.resize(triang.n_active_cells());
        lift_mats.resize(triang.n_active_cells());

        l_rhs.resize(triang.n_active_cells());
        for(auto &cur_rhs: l_rhs) cur_rhs.reinit(fe.dofs_per_cell);
}

/**
 * @brief Assembles the system
 * 
 * Calculating mass and differentiation matrices is as usual. Each face will have its own flux
 * matrix. The containers advection2D::face_first_dof and advection2D::face_dof_increment are used
 * to map face-local dof index to cell dof index.
 * 
 * @todo Check whether <code>FEFaceValues::reinit()</code> automatically takes care of direction of
 * integration for faces 0 and 3 since they are oriented in cw direction.
 * @remark Probably <code>FEFaceValues::reinit()</code> automatically takes care of direction of a
 * face because the code was working ok
 */
void advection2D::assemble_system()
{
        deallog << "Assembling system ... " << std::flush;
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
        // compute mass and diff matrices
        for(auto &cell: dof_handler.active_cell_iterators()){
                // deallog << "Assembling cell " << cell->index() << std::endl;
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

                // Lifting matrices for faces 0 and 3 must be muliplied by -1 (?)
                // not sure of this
                // lift_mats[cell->index()][0] *= -1.0;
                // lift_mats[cell->index()][3] *= -1.0;

        }// loop over cells
        deallog << "Completed assembly" << std::endl;
}

/**
 * @brief Sets initial condition
 * 
 * Since nodal basis is being used, initial condition is easy to set. interpolate function of
 * VectorTools namespace is used with IC class and advection2D::g_solution. See IC::value()
 */
void advection2D::set_IC()
{
        VectorTools::interpolate(dof_handler, IC(), g_solution);
}

/**
 * @brief Boundary ids are set here
 * 
 * @f$x=0@f$ forms boundary 0 with @f$\phi@f$ value prescribed as @f$1@f$<br/>
 * @f$y=0@f$ forms boundary 1 with @f$\phi@f$ value prescribed as @f$0@f$<br/>
 * @f$x=1 \bigcup y=1@f$ forms boundary 2 with zero gradient
 * @note Ghost cell approach will be used
 * @todo Check this function
 */
void advection2D::set_boundary_ids()
{
        for(auto &cell: dof_handler.active_cell_iterators()){
                for(uint face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        if(cell->face(face_id)->at_boundary()){
                                Point<2> fcenter = cell->face(face_id)->center(); // face center
                                if(fabs(fcenter(0)) < 1e-6)
                                        cell->face(face_id)->set_boundary_id(0);
                                else if(fabs(fcenter(1)) < 1e-6)
                                        cell->face(face_id)->set_boundary_id(1);
                                else
                                        cell->face(face_id)->set_boundary_id(2);
                        }
                } // loop over faces
        } // loop over cells
}

/**
 * @brief Updates solution with the given @p time_step
 * 
 * Algorithm:
 * - For every cell:
 *   - For every face:
 *     - Get neighbor id
 *     - if neighbor id > cell id, continue
 *     - else:
 *       - Get face id wrt owner and neighbor (using neighbor_of_neighbor)
 *       - Get global dofs on owner and neighbor
 *       - Using face ids and global dofs of owner and neighbor, get global dofs on this face on
 * owner and neighbor side
 *       - Compute the numerical flux
 *       - Use lifting matrices to update owner and neighbor rhs
 * 
 * <code>cell->get_dof_indices()</code> will return the dof indices in the order shown in
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html. This fact is mentioned in
 * https://www.dealii.org/current/doxygen/deal.II/classDoFCellAccessor.html.
 * To get the dof location, advection2D::dof_locations has been obtained using
 * <code>DoFTools::map_dofs_to_support_points()</code>. To get normal vectors, an FEFaceValues
 * object is created with Gauss-Lobatto quadrature of order <code>fe.degree+1</code>.
 * 
 * The face normal flux vector must be mapped to owner- and neighbor- local dofs for multplication
 * with lifting matrices. The mapped vectors will be of size <code>dof_per_cell</code>.
 * 
 * @pre @p time_step must be a stable one, any checks on this value are not done
 * @todo Some code repitition exists in the loop over faces
 */
void advection2D::update(const double time_step)
{
        // update old solution
        uint i;
        for(i=0; i<dof_handler.n_dofs(); i++) gold_solution(i) = g_solution(i);

        // set rhs to zero
        for(auto &cur_rhs: l_rhs) cur_rhs=0.0;

        uint face_id, face_id_neighbor; // id of face wrt owner and neighbor
        uint l_dof_id, l_dof_id_neighbor; // dof id (on a face) dof wrt owner and neighbor
        // global dof ids of owner and neighbor
        std::vector<uint> dof_ids(fe.dofs_per_cell), dof_ids_neighbor(fe.dofs_per_cell);
        double phi, phi_neighbor; // owner and neighbor side values of phi
        double cur_normal_flux; // normal flux at current dof
        // the -ve of normal num flux vector of face wrt owner and neighbor
        Vector<double> neg_normal_flux(fe.dofs_per_cell), neg_normal_flux_neighbor(fe.dofs_per_cell);
        Point<2> dof_loc; // dof coordinates (on a face)
        Tensor<1,2> normal; // face normal from away from owner at current dof
        FEFaceValues<2> fe_face_values(fe, QGaussLobatto<1>(fe.degree+1), update_normal_vectors);

        for(auto &cell: dof_handler.active_cell_iterators()){
                cell->get_dof_indices(dof_ids);
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        if(cell->face(face_id)->at_boundary()){
                                // this face is part of boundary, set phi_neighbor appropriately
                                fe_face_values.reinit(cell, face_id);
                                for(i=0; i<fe_face.dofs_per_face; i++){
                                        l_dof_id = face_first_dof[face_id] +
                                                i*face_dof_increment[face_id];
                                        
                                        normal = fe_face_values.normal_vector(i);
                                        // owner and neighbor side dof locations will match
                                        dof_loc = dof_locations[
                                                dof_ids[ l_dof_id ]
                                        ];

                                        phi = gold_solution[
                                                dof_ids[ l_dof_id ]
                                        ];
                                        // use array of functions (or func ptrs) to set BC
                                        phi_neighbor =
                                                bc_fns[cell->face(face_id)->boundary_id()](phi);

                                        cur_normal_flux = rusanov_flux(phi, phi_neighbor, dof_loc,
                                                normal);
                                        neg_normal_flux(l_dof_id) = -cur_normal_flux;
                                } // loop over face dofs

                                // multiply normal flux with lift matrx and store in rhs
                                lift_mats[cell->index()][face_id].vmult_add(
                                        l_rhs[cell->index()],
                                        neg_normal_flux
                                );
                        }
                        else if(cell->neighbor_index(face_id) > cell->index()) continue;
                        else{
                                // internal face
                                fe_face_values.reinit(cell, face_id);
                                face_id_neighbor = cell->neighbor_of_neighbor(face_id);
                                cell->neighbor(face_id)->get_dof_indices(dof_ids_neighbor);
                                for(i=0; i<fe_face.dofs_per_face; i++){
                                        l_dof_id = face_first_dof[face_id] +
                                                i*face_dof_increment[face_id];
                                        l_dof_id_neighbor = face_first_dof[face_id_neighbor] +
                                                i*face_dof_increment[face_id_neighbor];
                                        
                                        normal = fe_face_values.normal_vector(i);
                                        // owner and neighbor side dof locations will match
                                        dof_loc = dof_locations[
                                                dof_ids[ l_dof_id ]
                                        ];

                                        phi = gold_solution[
                                                dof_ids[ l_dof_id ]
                                        ];
                                        phi_neighbor = gold_solution[
                                                dof_ids_neighbor[ l_dof_id_neighbor ]
                                        ];

                                        cur_normal_flux = rusanov_flux(phi, phi_neighbor, dof_loc,
                                                normal);
                                        neg_normal_flux(l_dof_id) = -cur_normal_flux;
                                        neg_normal_flux_neighbor(l_dof_id_neighbor) = cur_normal_flux;
                                } // loop over face dofs

                                // multiply normal flux with lift matrx and store in rhs
                                // for both owner and neighbor
                                lift_mats[cell->neighbor_index(face_id)][face_id_neighbor].vmult_add(
                                        l_rhs[cell->neighbor_index(face_id)],
                                        neg_normal_flux_neighbor
                                );
                                lift_mats[cell->index()][face_id].vmult_add(
                                        l_rhs[cell->index()],
                                        neg_normal_flux
                                );
                        }
                } // loop over faces
                // compute stiffness term
                Vector<double> lold_solution(fe.dofs_per_cell); // old phi values of cell
                for(i=0; i<fe.dofs_per_cell; i++) lold_solution[i] = gold_solution[dof_ids[i]];
                stiff_mats[cell->index()].vmult_add(
                        l_rhs[cell->index()],
                        lold_solution
                );
        } // loop over cells (for assembling rhs)

        // Now, update
        for(auto &cell: dof_handler.active_cell_iterators()){
                cell->get_dof_indices(dof_ids);
                for(i=0; i<fe.dofs_per_cell; i++){
                        g_solution[dof_ids[i]] = gold_solution[dof_ids[i]] +
                                l_rhs[cell->index()][i] * time_step;
                }
        }
}

/**
 * @brief Prints stifness and the 4 lifting matrices of 0-th element
 */
void advection2D::print_matrices() const
{
        deallog << "Stiffness matrix" << std::endl;
        stiff_mats[0].print(deallog, 10, 2);
        for(uint i=0; i<GeometryInfo<2>::faces_per_cell; i++){
                deallog << "Lifting matrix, face " << i << std::endl;
                lift_mats[0][i].print(deallog, 15, 4);
        }
}

/**
 * @brief Outputs the global solution in vtk format taking the filename as argument
 */
void advection2D::output(const std::string &filename) const
{
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(g_solution, "phi");

        data_out.build_patches();

        std::ofstream ofile(filename);
        data_out.write_vtk(ofile);
}



// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
// Test function
// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#ifdef DEBUG
void advection2D::test()
{
        deallog << "---------------------------------------------" << std::endl;
        deallog << "Testing advection2D class" << std::endl;
        deallog << "---------------------------------------------" << std::endl;
        advection2D problem(1);
        problem.setup_system();
        problem.assemble_system();
        problem.print_matrices();
        problem.set_IC();
        problem.set_boundary_ids();

        double start_time = 0.0, end_time = 0.5, time_step = 0.005;
        uint time_counter = 0;
        std::string base_filename = "output.vtk";
        problem.output(base_filename + ".0"); // initial condition
        for(double cur_time = start_time; cur_time<end_time; cur_time+=time_step){
                deallog << "Step " << time_counter << " time " << cur_time << std::endl;
                problem.update(time_step);
                time_counter++;
                problem.output(base_filename + "." + std::to_string(time_counter));
        }
}
#endif
