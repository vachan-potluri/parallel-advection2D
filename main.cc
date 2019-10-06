/**
 * @file main.cc
 * @brief The main file of the project
 */

#include "advection2D.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        printf("Hello World!\n");
        #ifdef DEBUG
        advection2D::test();
        #endif
        return 0;
}