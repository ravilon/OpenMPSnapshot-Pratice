/*
  Copyright (C) 2020-2022 OMPTB Contributors

  This file is part of OMPTB.
 
  OMPTB is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OMPTB is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OMPTB.  If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
*/

/**
 * omp_bench.c
 *
 * @brief OMPTB benchmark harness entry point
 * @author Sascha Hunold 
 * @author Lukas Briem
 * @author Klaus Kraßnitzer
 * @date June 2020 - July 2022
 * @copyright GNU GPLv3
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <sys/stat.h>

#include "tasklib.h"
#include "util.h"
#include "hetero_util.h"

int RECORD_TRACE = 0;
int RECORD_NUM_TASKS = 0;

void print_usage(int argc, char *argv[]) {
    printf("%s \n"
           "-m tasks size (if homogeneous)\n"
           "-l lambda (if heterogeneous)\n"
           "-n number of task\n"
           "-s strategy (0=master, 1=parallel, 2=for)\n"
           "-v verbose mode\n"
           "-p number of threads\n"
           "-t trace output file\n"
           "-d data output file\n"
           "-r profiling (record number of tasks per thread)\n"
           "-o output file (statistics per thread)\n"
           "-j jedule trace\n",
           argv[0]);
}

int main(int argc, char *argv[]) {

    // number of threads
    int p = 1;

    // workload size
    long long n = 1; // number of tasks
    long long m = 1; // work per task

    // time recording
    double time_start, time_end;

    // trace recording 
    double *stimes, *etimes;
    double *tstart, *tspawned;
    int *tids;
    int *task_count;

    // heterogenous workloads
    double lambda = -1;
    const int workload_size = 10000;
    int *h_workloads = NULL;

    // argument parsing
    int tok;
    opterr = 0;

    // misc
    int i;
    bool hetero = false;
    bool data_file = false;
    char outfname[200];
    char jedfname[200];
    char thread_data_fname[200];
    char datafname[200];
    int write_trace = 0;
    int write_output = 0;
    int write_jedule = 0;
    int strategy = 0;
    int verbose = 0;
    double res;

    // parse arguments
    while ((tok = getopt(argc, argv, "n:m:p:t:s:o:j:l:d:vhr")) != -1) {
        switch (tok) {
            case 'n':
                n = atoll(optarg);
                break;
            case 'm':
                m = atoll(optarg);
                break;
            case 'p':
                p = atoi(optarg);
                break;
            case 'j':
                strncpy(jedfname, optarg, sizeof(outfname));
                write_jedule = 1;
                break;
            case 't':
                strncpy(outfname, optarg, sizeof(outfname));
                write_trace = 1;
                RECORD_TRACE = 1;
                break;
            case 's':
                strategy = atoi(optarg);
                break;
            case 'o':
                strncpy(thread_data_fname, optarg, sizeof(thread_data_fname));
                write_output = 1;
                break;
            case 'v':
                verbose = 1;
                break;
            case 'l':
                hetero = true;
                lambda = strtod(optarg, NULL);
                break;
            case 'r':
                RECORD_NUM_TASKS = 1;
                break;
            case 'd':
                strncpy(datafname, optarg, sizeof(datafname));
                data_file = true;
                break;
            case 'h':
                print_usage(argc, argv);
                exit(0);
            default:
                fprintf(stderr, "unknown parameter\n");
                print_usage(argc, argv);
                exit(1);
        }
    }


    if (verbose == 1) {

        printf("p=%d\n", p);
        printf("n=%lld\n", n);
        printf("s=%d\n", strategy);
        printf("m=%lld\n", m);

        double rtime, stime;
        int num_runs = 10000;
        stime = omp_get_wtime();
        for (i = 0; i < num_runs; i++) {
            res = add_bench(m);
        }
        rtime = omp_get_wtime() - stime;
        printf("kernel time (ms): %.6f\n", rtime / num_runs * 1e3);

        fflush(stdout);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(p);


    if (RECORD_TRACE) {
        tstart = malloc(p * sizeof(double));
        tspawned = malloc(p * sizeof(double));
        stimes = malloc(n * sizeof(double));
        etimes = malloc(n * sizeof(double));
        tids = malloc(n * sizeof(int));

        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            stimes[i] = 0.0;
            etimes[i] = 0.0;
            tids[i] = 0;
        }
    }

    if (RECORD_NUM_TASKS) {
        // sizeof(int) is redundant since CACHE_LINE_SIZE is given in bytes
        task_count = malloc(p*CACHE_LINE_SIZE);

        #pragma omp parallel for
        for (i=0; i<p; i++) {
            task_count[FS_OFFSET*i] = 0;
        }
    }

    // record start time
    time_start = omp_get_wtime();

    /**
     * @brief Runs the benchmark with heterogeneous workloads, i.e., generates a workload out of exponential distribution
     * and draw from it every iteration.
     * Lambda parameter of exponential distribution can be specified with option -l
     *
     */
    if (hetero) {
        // initial heterogenous workload from exponential distribution
        //srand(time(NULL) * getpid());
        srand(10);
        h_workloads = expo_dist(lambda, workload_size);
    }

    if (strategy == 0) {
        // Master Strategy

        #pragma omp parallel firstprivate(m)
        {
            if (RECORD_TRACE) {
                tstart[omp_get_thread_num()] = omp_get_wtime();
            }

            #pragma omp master
            {
                for (i = 0; i < n; i++) {

                    if (hetero)
                        m = h_workloads[(i % workload_size)];

                    #pragma omp task shared(stimes, etimes, tids, res) firstprivate(i, m)
                    {

                        if (RECORD_TRACE) {
                            stimes[i] = omp_get_wtime();
                            tids[i] = omp_get_thread_num();
                        }

                        if (RECORD_NUM_TASKS) {
                            task_count[omp_get_thread_num()*FS_OFFSET]++;
                        }

                        res = add_bench(m);

                        if (RECORD_TRACE) {
                            etimes[i] = omp_get_wtime();
                        }
                    }
                }
            }
            if (RECORD_TRACE) {
                tspawned[omp_get_thread_num()] = omp_get_wtime();
            }
            #pragma omp taskwait
        }

    } else if (strategy == 1) {
        // Parallel Strategy

        #pragma omp parallel firstprivate(m)
        {
            if (RECORD_TRACE) {
                tstart[omp_get_thread_num()] = omp_get_wtime();
            }

            #pragma omp for
            for (i = 0; i < n; i++) {

                if (hetero)
                    m = h_workloads[(i % workload_size)];

                #pragma omp task shared(stimes, etimes, tids, res) firstprivate(i, m)
                {

                    if (RECORD_TRACE) {
                        stimes[i] = omp_get_wtime();
                        tids[i] = omp_get_thread_num();
                    }

                    if (RECORD_NUM_TASKS) {
                        task_count[omp_get_thread_num()*FS_OFFSET]++;
                    }

                    res = add_bench(m);

                    if (RECORD_TRACE) {
                        etimes[i] = omp_get_wtime();
                    }
                }
            }

            if (RECORD_TRACE) {
                tspawned[omp_get_thread_num()] = omp_get_wtime();
            }
            #pragma omp taskwait
        }

    } else if (strategy == 2) {
        // Parallel For Strategy 

        #pragma omp parallel firstprivate(m)
        {
            if (RECORD_TRACE) {
                tstart[omp_get_thread_num()] = omp_get_wtime();
            }

            #pragma omp for
            for (i = 0; i < n; i++) {

                if (hetero)
                    m = h_workloads[(i % workload_size)];

                if (RECORD_TRACE) {
                    stimes[i] = omp_get_wtime();
                    tids[i] = omp_get_thread_num();
                }

                if (RECORD_NUM_TASKS) {
                    task_count[omp_get_thread_num()*FS_OFFSET]++;
                }

                res = add_bench(m);

                if (RECORD_TRACE) {
                    etimes[i] = omp_get_wtime();
                }
            }

            if (RECORD_TRACE) {
                tspawned[omp_get_thread_num()] = omp_get_wtime();
            }
        }

    } else {
        fprintf(stderr, "unknown strategy %d\n", strategy);
        exit(1);
    }

    if (hetero) {
        free(h_workloads);
    }

    // record end time
    time_end = omp_get_wtime();

    // write output
    FILE *outf = data_file? fopen(datafname, "w") : stdout;

    if (hetero) {
        fprintf(outf, "p;n;l;s;time\n");
        fprintf(outf, "%d;%lld;%g;%d;%.6f\n", p, n, lambda, strategy, time_end - time_start);
    } else {
        fprintf(outf, "p;n;m;s;time\n");
        fprintf(outf, "%d;%lld;%lld;%d;%.6f\n", p, n, m, strategy, time_end - time_start);
    }

    if (RECORD_TRACE) {
        // write trace output data

        double rtime = 0;

        for (i = 0; i < n; i++) {
            rtime += etimes[i] - stimes[i];
        }

        if (verbose) {
            printf("mean task time [mus]: %10.6f\n", rtime*1e6/n);

            for(i=0; i<p; i++) {
            printf("%4d: task_count: %8d\n", i, count_tid[i]);
            }
        }

        struct omp_trace_data trace_data = {
            p, n, m, lambda, strategy,
            time_start,
            time_end,
            rtime * 1e3 / n,
            stimes,
            etimes,
            tids
        };

        if (write_trace == 1) {
            write_events_csv(outfname, &trace_data, hetero);
        }

        if (write_jedule == 1) {
            write_trace_file(jedfname, &trace_data);
        }


        struct omp_stats_data omp_stats_data = {
                &trace_data,
                tstart,
                tspawned,
        };

        if (write_output == 1) {
            write_thread_csv(thread_data_fname, &omp_stats_data, hetero);
        } else {
            write_thread_csv("/dev/stdout", &omp_stats_data, hetero);
        }


        free(stimes);
        free(etimes);
        free(tids);
    }

    if (RECORD_NUM_TASKS) {
        if (write_output) {
            write_profile_csv(thread_data_fname, task_count, p, n, m, lambda, hetero);
        } else {
            write_profile_csv("/dev/stdout", task_count, p, n, m, lambda, hetero);
        }
    }

    return 0;
}
