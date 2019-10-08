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
{
        MPI_Barrier(mpi_communicator);
}

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

        g_solution.reinit(locally_owned_dofs, mpi_communicator);
        gold_solution.reinit(locally_owned_dofs, mpi_communicator);
        g_rhs.reinit(locally_owned_dofs, mpi_communicator);
        gh_gold_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

        MPI_Barrier(mpi_communicator);
}

/**
 * @brief Assembles the system
 * 
 * Calculating mass and differentiation matrices is as usual. Each face will have its own flux
 * matrix. The containers advection2D::face_first_dof and advection2D::face_dof_increment are used
 * to map face-local dof index to cell dof index.
 * 
 * @note The matrices for ghost cells are also computed and stored. This is required for update.
 * Also, all cell indexing for an individual mpi process is local. The cell's index and its
 * neighbors indices are locally defined.
 * 
 * @todo See if there is another approach. For example, if the numerical flux is stored as a global
 * vector, then every mpi process can update without having to know ghost cell matrices.
 * 
 * @remark Storing matrices for ghost cells too is the best approach. The issues with above approach
 * of storing and communicating numerical flux is that mapping from face dofs to cell dofs must be
 * done repeatedly and the sign of the numerical flux changes from cell to cell based on how it was
 * calculated.
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
                // skip if cell is not relevant to this mpi proc (owned + ghost)
                if(!(cell->is_locally_owned() || cell->is_ghost())) continue;

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
        } // loop over locally relevant cells

        MPI_Barrier(mpi_communicator);
}

/**
 * @brief Sets IC
 * 
 * Since nodal basis is being used, initial condition is easy to set. <code>interpolate</code>
 * function of VectorTools namespace is used with IC class and advection2D::g_solution.
 * See IC::value()
 * 
 * @note VectorTools::interpolate can only be used on vectors with no ghost cells.
 */
void advection2D::set_IC()
{
        VectorTools::interpolate(dof_handler, IC(), g_solution);

        // MPI_Barrier(mpi_communicator); // not required, this is a collective operation
}

/**
 * @brief Boundary ids are set here
 * 
 * @f$x=0@f$ forms boundary 0 with @f$\phi@f$ value prescribed as @f$1@f$<br/>
 * @f$y=0@f$ forms boundary 1 with @f$\phi@f$ value prescribed as @f$0@f$<br/>
 * @f$x=1 \bigcup y=1@f$ forms boundary 2 with zero gradient
 * @note Ghost cell approach will be used
 * @note Boundary ids must be set strictly for appropriate faces of owned cells only
 */
void advection2D::set_boundary_ids()
{
        for(auto &cell: dof_handler.active_cell_iterators()){
                // skip if the cell is not owned
                if(!(cell->is_locally_owned())) continue;

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
        } // loop over locally owned cells

        MPI_Barrier(mpi_communicator);
}

/**
 * @brief Obtains time step based on Courant number @p co
 * 
 * If @f$r@f$ is the "radius" of the cell, then
 * @f[
 * \Delta t = \text{Co}\frac{1}{2N+1}\min\left[ \frac{r}{u}, \frac{r}{v} \right]
 * @f]
 */
void advection2D::obtain_time_step(const double co)
{
        double radius,  // radius of cell
                proc_min=1e6, // (factor of) time step for this mpi process
                min, // (factor of) time step after reduction
                temp;
        Tensor<1,2> center_wind;
        for(auto &cell: dof_handler.active_cell_iterators()){
                // skip if cell is not owned by this process
                if(!(cell->is_locally_owned())) continue;

                radius = 0.5*cell->diameter();
                center_wind = wind(cell->center());
                temp = std::min({
                        radius/(center_wind[0]+1e-6),
                        radius/(center_wind[1]+1e-6)
                });

                if(temp<proc_min) proc_min = temp;
        } // loop over locally owned cells
        MPI_Barrier(mpi_communicator);

        // first perform reduction (into min of 0-th process)
        MPI_Reduce(&proc_min, &min, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_communicator);

        // now multiply by factor and broadcast
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
                min *= co/(2*fe.degree + 1);
        }
        MPI_Bcast(&min, 1, MPI_DOUBLE, 0, mpi_communicator);

        time_step = min;
        std::cout << "Process " << Utilities::MPI::this_mpi_process(mpi_communicator) <<
        " time step " << time_step << std::endl;
}

/**
 * @brief Updates the solution taking advection2D::time_step as time step
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
 * @note Here, ghosted copy of old solution is used for accessing ghost values.
 * @todo This function might have some serious algo issues
 */
void advection2D::update()
{
        // update old solution and make a ghost copy of it
        gold_solution = g_solution;
        gh_gold_solution = gold_solution;
        g_rhs = 0.0;

        uint i;
        uint face_id, face_id_neighbor; // id of face wrt owner and neighbor
        uint l_dof_id, l_dof_id_neighbor; // dof id (on a face) dof wrt owner and neighbor
        // global dof ids of owner and neighbor
        std::vector<uint> dof_ids(fe.dofs_per_cell), dof_ids_neighbor(fe.dofs_per_cell);
        double phi, phi_neighbor; // owner and neighbor side values of phi
        double cur_normal_flux; // normal flux at current dof
        // the -ve of normal num flux vector of face wrt owner and neighbor
        Vector<double> neg_normal_flux(fe.dofs_per_cell), neg_normal_flux_neighbor(fe.dofs_per_cell);
        // rhs vectors for owner and neighbor
        // contrib from rhs_neighbor will be added through compress operation
        Vector<double> rhs(fe.dofs_per_cell), rhs_neighbor(fe.dofs_per_cell);
        Point<2> dof_loc; // dof coordinates (on a face)
        Tensor<1,2> normal; // face normal from away from owner at current dof
        // fe face values initialised with Lobatto points to obtain normals at quad points
        FEFaceValues<2> fe_face_values(fe, QGaussLobatto<1>(fe.degree+1), update_normal_vectors);

        for(auto &cell: dof_handler.active_cell_iterators()){
                // skip if not locally owned
                if(!(cell->is_locally_owned())) continue;

                // unlike rhs, rhs_neighbor is not affiliated to a single cell
                rhs = 0.0;
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

                                // multiply normal flux with lift matrx and add to owners rhs
                                lift_mats[cell->index()][face_id].vmult_add(
                                        rhs,
                                        neg_normal_flux
                                );
                        } // if boundary face
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
                                lift_mats[cell->neighbor_index(face_id)][face_id_neighbor].vmult(
                                        rhs_neighbor,
                                        neg_normal_flux_neighbor
                                );
                                for(i=0; i<fe.dofs_per_cell; i++){
                                        // accessing data of other process
                                        g_rhs[dof_ids_neighbor[i]] = rhs_neighbor[i];
                                }
                                lift_mats[cell->index()][face_id].vmult_add(
                                        rhs,
                                        neg_normal_flux
                                );
                        } // if internal face
                } // loop over faces
                g_rhs.compress(VectorOperation::add);

                // compute stiffness term
                Vector<double> lold_solution(fe.dofs_per_cell); // old phi values of cell
                for(i=0; i<fe.dofs_per_cell; i++) lold_solution[i] = gold_solution[dof_ids[i]];
                stiff_mats[cell->index()].vmult_add(
                        rhs,
                        lold_solution
                ); // rhs now has all owned face contribs and stiffness contrib
                for(i=0; i<fe.dofs_per_cell; i++) g_rhs[dof_ids[i]] += rhs[i]; // non-ghost access
                // set owners rhs using dof_ids
                // set neighbors rhs and compress
        } // loop over locally owned cells (computing rhs)
}


/**
 * @brief Prints stifness and the 4 lifting matrices of 0-th element of 0-th process
 */
void advection2D::print_matrices() const
{
        pcout << "Stiffness matrix and lifting matrices of the first cell owned by " <<
        "first mpi process" << std::endl;

        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
                uint row, col, face_id;
                pcout << std::fixed << std::setprecision(6);

                // find the first owned cell of this process
                DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();
                for(; cell != dof_handler.end(); cell++){
                        if(cell->is_locally_owned()) break;
                }

                // stiffness matrix
                pcout << "Stiffness matrix" << std::endl;
                for(row=0; row<fe.dofs_per_cell; row++){
                        for(col=0; col<fe.dofs_per_cell; col++){
                                // operator [] is non-const and cannot be used in a const function
                                pcout << std::setw(12) << stiff_mats.at(cell->index())(row,col);
                        }
                        pcout << std::endl;
                }

                // lifting matrices
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        pcout << "Lifting matrix face " << face_id << std::endl;
                        for(row=0; row<fe.dofs_per_cell; row++){
                                for(col=0; col<fe.dofs_per_cell; col++){
                                        pcout << std::setw(12) <<
                                        lift_mats.at(cell->index())[face_id](row,col);
                                }
                                pcout << std::endl;
                        }
                }
        }
}

/**
 * @brief Outputs the global solution in pvtu format taking the filename as argument
 * 
 * @precondition @p filename must not have extension. Checks in this regard are not done
 */
void advection2D::output(const std::string &filename) const
{
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(g_solution, "phi");

        Vector<float> subdom(triang.n_active_cells());
        for(float &x: subdom){
                x = triang.locally_owned_subdomain();
        }
        data_out.add_data_vector(subdom, "Subdomain");

        data_out.build_patches();

        // Inidividual process solution files
        std::string mod_filename(filename);
        mod_filename += "_" + Utilities::int_to_string(triang.locally_owned_subdomain(), 2) + ".vtu";
        std::ofstream ofile(mod_filename);
        data_out.write_vtu(ofile);

        // master output file
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
                std::vector<std::string> filenames;
                for(uint i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); i++){
                        filenames.emplace_back(filename + "_" +
                                Utilities::int_to_string(i, 2) + ".vtu");
                }
                std::ofstream master(filename + ".pvtu");
                data_out.write_pvtu_record(master, filenames);
        }
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
        MPI_Barrier(MPI_COMM_WORLD);
        problem.print_matrices();
        problem.set_IC();
        problem.set_boundary_ids();
        problem.output("partition");
        problem.obtain_time_step(0.3);

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
