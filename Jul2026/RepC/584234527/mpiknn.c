/* mpiknn - compute the all-knn problem
 * Copyright (C) 2023  Alexandros Athanasiadis
 *
 * This file is part of knn
 *
 * knn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * knn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <string.h>
#include <errno.h>
#include <time.h>

#include <math.h>

#include <mpi.h>
#include <omp.h>

#include "knn.h"
#include "matrix.h"
#include "def.h"

/* MPI constants and tags */
#define ROOT 0

#define INIT_MATRIX 1
#define KNN 2
#define PRINTING_DIST 3
#define PRINTING_IDX 4

/* function that reads n * d elements from file into X */
int read_matrix(FILE *file, elem_t *X, size_t n, size_t d) {
	for(size_t i = 0 ; i < n ; i++) {
		for(size_t j = 0 ; j < d ; j++) {
			int k = fscanf(file, "%f", &MATRIX_ELEM(X, i, j, n, d));

			if(!k && feof(file)) {
				return 1;
			}
		}
	}

	return 0;
}

/* mpiknn calculates all-knn for all input points using MPI on a
 * set of distributed compute nodes.
 *
 * The input array is split equally between all the nodes then each
 * node is responsible for finding the k nearest neighbours of all
 * of the points in its corresponding subset.
 *
 * the root process is responsible for I/O and initialization as
 * well as error checking.
 */
int main(int argc, char **argv) {
	/* initialize MPI environement */
	MPI_Init(&argc, &argv);

	/* get number of processes and the process rank */
	int n_processes, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* initialize variables */
	int error = 0;

	FILE *input_file = NULL;
	FILE *output_file = NULL;
	FILE *log_file = NULL;

	size_t N, d, k = 0;
	size_t max_mem = 1*GB;

	/* parse command line options and arguments */
	switch(rank) {
		case ROOT: ;
			char *input_fname = NULL;
			char *output_fname = NULL;
			char *log_fname = NULL;

			error = 0;

			/* use getopt to parse command line options */
			int c;
			while((c = getopt(argc, argv, "i:o:l:k:m:")) != -1) {
				switch(c) {
					/* option -i specifies the input device */
					case 'i':
						input_fname = optarg;
						break;
						
					/* option -o specifies the output device */
					case 'o':
						output_fname = optarg;
						break;

					/* option -l specifies the log file */
					case 'l':
						log_fname = optarg;
						break;

					/* option -k specifies the number of nearest neighbours */
					case 'k':
						k = atoi(optarg);
						break;

					/* option -m specifies the memory to use per process */
					case 'm': ;
						size_t size;
						char unit; 

						/* read a size and a unit multiplier */
						int read_count = sscanf(optarg, "%zu%c", &size, &unit);

						size_t unit_size = 1;
						if(read_count == 0) {
							error = EINVAL;
							fprintf(stderr, "error parsing parameter '%c': %s\n", 'm', strerror(error));
							break;

						} else if(read_count == 2) {
							/* use the correct unit multiplier */
							switch(unit) {
								case 'k':
								case 'K':
									unit_size = KB;
									break;

								case 'm':
								case 'M':
									unit_size = MB;
									break;

								case 'g':
								case 'G':
									unit_size = GB;
									break;

								default:
									error = EINVAL;
									fprintf(stderr, "error parsing parameter '%c': %s\n", 'm', strerror(error));
									break;
							}
						}

						/* calculate the size in bytes */
						max_mem = size * unit_size;
						break;

					case '?':
						error = EINVAL;
						fprintf(stderr, "error: parameter '%c' requires an input argument\n", optopt);
						break;
				}
			}

			if(error) break;

			/* if input device name was not specified using -i use argv[optind]
			 * or if that is also unspecified then stdin*/
			if(input_fname == NULL) {
				if(optind < argc) {
					input_fname = argv[optind++];
				} else {
					input_fname = "stdin";
					input_file = stdin;
				}
			}

			/* if output device name was not specified using -o use stdout */
			if(output_fname == NULL) {
				output_fname = "stdout";
				output_file = stdout;
			}

			/* if log file name was not specified using -l use stdout */
			if(log_fname == NULL) {
				log_fname = "stdout";
				log_file = stdout;
			}

			/* if input file has not been opened, open input file to input file name */
			if((input_file == NULL) && (input_file = fopen(input_fname, "r")) == NULL) {
				error = errno;
				fprintf(stderr, "%s: %s\n", input_fname, strerror(error));
				break;
			}

			/* if output file has not been opened, open output file to output file name */
			if((output_file == NULL) && (output_file = fopen(output_fname, "w")) == NULL) {
				error = errno;
				fprintf(stderr, "%s: %s\n", output_fname, strerror(error));
				break;
			}

			/* if log file has not been opened, open log file to log file name */
			if((log_file == NULL) && (log_file = fopen(log_fname, "a")) == NULL) {
				error = errno;
				fprintf(stderr, "%s: %s\n", log_fname, strerror(error));
				break;
			}

			/* if k was unspecified set k from argv[optind] */
			if(k == 0) {
				if(optind >= argc) {
					error = EINVAL;
					fprintf(stderr, "error: not enough input arguments\n");
					break;
				} else {
					k = atoi(argv[optind++]);
				}
			}

			/* read matrix dimensions from input device */
			int count = fscanf(input_file, "%zu %zu\n", &N, &d);
			if(count < 2) {
				error = errno;
				fprintf(stderr, "error reading from %s: %s\n", input_fname, strerror(error));
				break;
			}

			break;
	}

	/* check for errors in options and arguments */
	MPI_Bcast(&error, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if(error) {
		MPI_Finalize();
		exit(error);
	}

	/* broadcast parameters to other processes */
	MPI_Bcast(&N, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&d, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&k, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&max_mem, 1, MPI_SIZE_T, ROOT, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	
	/* n_max is the max number of rows that any process can have*/
	size_t n_max = (size_t)ceil((double)N/(double)n_processes);

	/* r_send and r_recv are initialized using the custom modulus function
	 * in order to get the previous and the next process in the ring */
	int r_send = mod(rank - 1, n_processes);
	int r_recv = mod(rank + 1, n_processes);

	/* set OpenMP max threads */
	int n_threads = omp_get_max_threads();
	omp_set_num_threads(n_threads);

	/* initialize X,Y and Z matrices */
	elem_t *X = create_matrix(n_max, d);
	elem_t *Y = create_matrix(n_max, d);
	elem_t *Z = create_matrix(n_max, d);

	/* calculate the begin and end indices of the subset of the input
	 * that corresponds to this process */
	int i_begin = rank * N / n_processes;
	int i_end = min((rank + 1) * N / n_processes, N);

	int n = i_end - i_begin;

	/* formula to get the maximum value of t for the specified memory size */
	intmax_t t = (intmax_t)(max_mem/n_max - 3*d*sizeof(elem_t) - 2*k*(sizeof(elem_t) + sizeof(size_t))) 
			 / (intmax_t)(sizeof(elem_t) + sizeof(size_t));

	/* then set t as the minimum between itself and n, since there is no point in t being larger than n */
	t = min(t, n);

	MPI_Barrier(MPI_COMM_WORLD);

	if(t < 1) {
		if(rank == ROOT) fprintf(stderr, "not enough memory.\n");
		MPI_Finalize();
		exit(ENOMEM);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	/* initialize the matrices */
	
	int recv_size = 0;
	MPI_Request send_req = MPI_REQUEST_NULL, recv_req;
	MPI_Status send_status, recv_status;

	switch(rank) {
		case ROOT: ;
			/* root process reads its own data from the input file */
			read_matrix(input_file, X, n, d);

			/* then it reads from the input file and sends 
			 * data to the corresponding process */
			for(int r = 1 ; r < n_processes ; r++) {
				int r_begin = r * N / n_processes;
				int r_end = min((r + 1) * N / n_processes, N);

				int r_n = r_end - r_begin;

				read_matrix(input_file, Y, r_n, d);

				MPI_Wait(&send_req, &send_status);
				MPI_Isend(Y, r_n * d, MPI_ELEM_T, r, INIT_MATRIX, MPI_COMM_WORLD, &send_req);

				SWAP(Y, Z);
			}

			if(input_file != stdin) fclose(input_file);

			break;

		default: ;
			/* other processes receive their data from the root process */
			MPI_Recv(X, n_max * d, MPI_ELEM_T, ROOT, INIT_MATRIX, MPI_COMM_WORLD, &recv_status);

			break;
	}

	/* also initialize Y = X */
	int m = n;
	#pragma omp parallel for simd
	for(size_t i = 0 ; i < m * d ; i++) {
		Y[i] = X[i];
	}
	
	/* then wait for the last send to finish before continuing */
	if(rank == ROOT) MPI_Wait(&send_req, &send_status);

	MPI_Barrier(MPI_COMM_WORLD);

	struct timespec t_begin, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_begin);

	/* perform knn */
	knn_result *res = NULL;

	for(int q = 0 ; q < n_processes ; q++) {
		/* initiate non-blocking send and recieve */
		MPI_Isend(Y, m * d, MPI_ELEM_T, r_send, KNN, MPI_COMM_WORLD, &send_req);
		MPI_Irecv(Z, n_max * d, MPI_ELEM_T, r_recv, KNN, MPI_COMM_WORLD, &recv_req);

		/* calculate the appropriate Y_idx */
		int r = mod(rank + q, n_processes);
		size_t Y_idx = r * N / n_processes;

		/* perform the knn operation */
		res = knn(X, n, Y, Y_idx, m, d, k, t, &res);

		/* wait for send and recieve to complete */
		MPI_Wait(&send_req, &send_status);
		MPI_Wait(&recv_req, &recv_status);

		MPI_Get_count(&recv_status, MPI_ELEM_T, &recv_size);
		m = recv_size / d;

		/* swap Y and Z pointer locations*/
		SWAP(Y, Z);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_MONOTONIC, &t_end);

	/* free matrices */
	delete_matrix(X);
	delete_matrix(Y);
	delete_matrix(Z);

	/* print results */
	switch(rank) {
		case ROOT: ;
			/* print information to log file */
			double time_elapsed = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_nsec - t_begin.tv_nsec) / 1e9f;
			fprintf(log_file, "%zu %zu %zu %zu %d %d %zu %zd %lf\n", N, d, k, max_mem, n_processes, n_threads, n_max, t, time_elapsed);
			
			/* create arrays to be used for printing */
			elem_t *dists= (elem_t *) malloc(n_max * k * sizeof(elem_t));
			elem_t *dists_swp = (elem_t *) malloc(n_max * k * sizeof(elem_t));

			size_t *idxs= (size_t *) malloc(n_max * k * sizeof(size_t));
			size_t *idxs_swp = (size_t *) malloc(n_max * k * sizeof(size_t));

			#pragma omp parallel for simd
			for(int i = 0 ; i < n * k ; i++) {
				dists[i] = res->n_dist[i];
				idxs[i] = res->n_idx[i];
			}

			MPI_Request recv_dist, recv_idx;
			MPI_Status dist_status, idx_status;

			/* root process recieves the results of the other 
			 * processes and prints them one by one */
			for(int r = 1 ; r < n_processes ; r++) {
				MPI_Irecv(dists_swp, n_max * k, MPI_ELEM_T, r, PRINTING_DIST, MPI_COMM_WORLD, &recv_dist);
				MPI_Irecv(idxs_swp, n_max * k, MPI_SIZE_T, r, PRINTING_IDX, MPI_COMM_WORLD, &recv_idx);

				for(size_t i = 0 ; i < n ; i++) {
					for(size_t j = 0 ; j < k ; j++) {
						fprintf(output_file, "%zu:%0.2f ", 
								MATRIX_ELEM(idxs, i, j, n, k), MATRIX_ELEM(dists, i, j, n, k));
					}
					fprintf(output_file, "\n");
				}

				MPI_Wait(&recv_dist, &dist_status);
				MPI_Wait(&recv_idx, &idx_status);

				MPI_Get_count(&dist_status, MPI_ELEM_T, &recv_size);

				n = recv_size / k;

				SWAP(dists, dists_swp);
				SWAP(idxs, idxs_swp);
				
			} for(size_t i = 0 ; i < n ; i++) {
				for(size_t j = 0 ; j < k ; j++) {
					fprintf(output_file, "%zu:%0.2f ", 
							MATRIX_ELEM(idxs, i, j, n, k), MATRIX_ELEM(dists, i, j, n, k));
				}
				fprintf(output_file, "\n");
			}

			if(output_file != stdout) fclose(output_file);
			if(log_file != stdout) fclose(log_file);

			/* free printing arrays */
			free(dists_swp);
			free(idxs_swp);

			break;

		default: ;
			/* other processes send their data to the root process */
			MPI_Isend(res->n_dist, n * k, MPI_ELEM_T, ROOT, PRINTING_DIST, MPI_COMM_WORLD, &send_req);
			MPI_Send(res->n_idx, n * k, MPI_SIZE_T, ROOT, PRINTING_IDX, MPI_COMM_WORLD);
			MPI_Wait(&send_req, &send_status);

			break;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* free knn results */
	delete_knn(res);

	MPI_Finalize();
}

