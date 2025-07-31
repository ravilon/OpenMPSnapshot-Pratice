/*
* Implementation file for FDM algorithms
*/

#include <iostream> // cout, DEBUGGING ONLY
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include "error_codes.hpp" // namespace err

/**
* solver_1
* Brute-force implementation of solving the heat equation on a
* massless wire. This follows the Python implementation.
* - Allocate memory for an TxN "matrix", with T timesteps and N
*    nodes.
* - Initialize boundary condition at left side of wire for all time
*    steps as 100 temperature units; this is a constant.
* - Create a double for-loop to iterate over time steps, and within
*    that, over nodes
* - Proceed with finite difference method node-by-node.
*
* ***** NOTES *****
* I use std::vector here to approximate the overhead of the Python
* script using a NumPy ND-array. This may be a poor approximation.
*
* Assumption: boundary condition at left side of wire at 100
* temperature units is a constant; everything else is initialized
* at 0.
*
*/
err::eErrorCode solver_1(
    int n_timesteps,
    std::vector<double> & temps_n,
    std::vector<double> & temps_n_plus_1,
    int n_nodes,
    float amp,
    std::string & data_file_prefix
) {

#if defined WRITEOUT && WRITEOUT==1
    // create "format string" (C++17 friendly) used for output file
    // TODO: use stringstream instead
    std::string fname = data_file_prefix + "_" + std::to_string(n_timesteps) + "_" + std::to_string(n_nodes) + ".out.bin";

    // create output file buffer to write binary data to; open for appending
    // so we don't have to create as many files
    std::ofstream outbuf(fname, std::ios::binary | std::ios::app);
#endif

    for (int i = 0; i<n_timesteps; ++i) {

        for (int j = 0; j<n_nodes - 1; ++j) {

#if defined DEBUG && DEBUG==1
            std::cout << std::fixed << std::setprecision(7) << (temps_n)[j] << " ";
#endif

            // left boudnary condition simply assigns the same value
            if (j==0) { temps_n_plus_1[j] = temps_n[j]; }

            // right boundary condition
            else if (i==n_nodes-1) {
                temps_n_plus_1[j] = amp * (temps_n[j-2] - 2*temps_n[j-1] + temps_n[j]);
            }

            else {
                temps_n_plus_1[j] = amp * (temps_n[j-1] - 2*temps_n[j] + temps_n[j+1]);
            }

        } // end node iteration

#if defined WRITEOUT && WRITEOUT==1
        // push to buffer
        outbuf.write(reinterpret_cast<char *>(temps_n_plus_1.data()), sizeof(double)*n_nodes);
        outbuf.flush(); // sync to filesystem
#endif

#if defined DEBUG && DEBUG==1
        std::cout << "\n";
#endif

        // swap vectors to keep cycling through
        temps_n.swap(temps_n_plus_1);

    } // end timestep iteration

#if defined WRITEOUT && WRITEOUT==1
    // close output file stream
    outbuf.close();
#endif

#if defined DEBUG && DEBUG==1
    std::flush(std::cout);
#endif


    return err::SUCCESS;

}


/**
* solver 2
* Nearly identical to sovler_1, save for using OpenMP to parallelize the computation
* across the nodes.
*/
err::eErrorCode solver_2(
    int n_timesteps,
    std::vector<double> & temps_n,
    std::vector<double> & temps_n_plus_1,
    int n_nodes,
    float amp,
    std::string & data_file_prefix
) {

#if defined WRITEOUT && WRITEOUT==1
    // create "format string" (C++17 friendly) used for output file
    // TODO: use stringstream instead
    std::string fname = data_file_prefix + "_" + std::to_string(n_timesteps) + "_" + std::to_string(n_nodes) + ".out.bin";

    // create output file buffer to write binary data to; open for appending
    // so we don't have to create as many files
    std::ofstream outbuf(fname, std::ios::binary | std::ios::app);
#endif

    // create indexing variables to use for iterating through timesteps
    // and nodes; define them here so they can be local to each thread
    // created using OpenMP
    int i, j;
    i = j = 0;

    // create threads with local indices i, j
    #pragma omp parallel private (i, j)
    {
        for (i = 0; i<n_timesteps; ++i) {

            // decompose iterations of parallel for loop into one contiguous block
            // per thread
            #pragma omp for schedule(static)
            for (j = 0; j<n_nodes - 1; ++j) {

#if defined DEBUG && DEBUG==1
                std::cout << std::fixed << std::setprecision(7) << (temps_n)[j] << " ";
#endif

                // left boudnary condition simply assigns the same value
                if (j==0) { temps_n_plus_1[j] = temps_n[j]; }

                // right boundary condition
                else if (i==n_nodes-1) {
                    temps_n_plus_1[j] = amp * (temps_n[j-2] - 2*temps_n[j-1] + temps_n[j]);
                }

                else {
                    temps_n_plus_1[j] = amp * (temps_n[j-1] - 2*temps_n[j] + temps_n[j+1]);
                }

            } // end node iteration

#if defined DEBUG && DEBUG==1
            #pragma omp single // only single thread can flush buffer
            std::cout << "\n";
            std::flush(std::cout);
#endif

#if defined WRITEOUT && WRITEOUT==1

            // push to buffer; NOTE must enforce single write/flush
            #pragma omp single
            outbuf.write(reinterpret_cast<char *>(temps_n_plus_1.data()), sizeof(double)*n_nodes);
            outbuf.flush(); // sync to filesystem
#endif

            // swap vectors to keep cycling through
            #pragma omp single // only one thread can act
            temps_n.swap(temps_n_plus_1);

        } // end timestep iteration

    } // end parallel private block

    //
    // back to single-threaded program
    //

#if defined WRITEOUT && WRITEOUT==1
    // close output file stream
    outbuf.close();
#endif

    return err::SUCCESS;

}

