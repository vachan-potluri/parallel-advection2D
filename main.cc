/**
 * @file main.cc
 * @brief The main file of the project
 */

#include "advection2D.h"

#include <iostream>
#include <fstream>

int main()
{
        deallog.depth_console(2);

        printf("Hello World!\n");
        #ifdef DEBUG
        advection2D::test();
        #endif
        return 0;
}