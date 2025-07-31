 // This class computes the contributions to the C_ab integral iterating over cells and particles. It is a heavily modified version of code by Alex Wiegand.

#ifndef COMPUTE_INTEGRAL_H
#define COMPUTE_INTEGRAL_H

#include "integrals.h"

class compute_integral{

    private:
        uint64 cnt4=0;
        int nbin, mbin;


    public:
        int particle_list(int id_1D, Particle* &part_list, int* &id_list, Grid *grid){
            // function updates a list of particles for a 1-dimensional ID. Output is number of particles in list.

            Cell cell = grid->c[id_1D]; // cell object
            int no_particles = 0;
            // copy in list of particles into list
            for (int i = cell.start; i<cell.start+cell.np; i++, no_particles++){
                part_list[no_particles]=grid->p[i];
                id_list[no_particles]=i;
            }
        return no_particles;
        }

    private:
        int draw_particle(integer3 id_3D, Particle &particle, int &pid, Float3 shift, Grid *grid, int &n_particles, gsl_rng* locrng, int &n_particles1, int &n_particles2){
            // Draw a random particle from a cell given the cell ID.
            // This updates the particle and particle ID and returns 1 if error.

            int id_1D = grid-> test_cell(id_3D);
            if(id_1D<0) return 1; // error if cell not in grid
            Cell cell = grid->c[id_1D];
            if(cell.np==0) return 1; // error if empty cell
            pid = floor(gsl_rng_uniform(locrng)*cell.np) + cell.start; // draw random ID
            particle = grid->p[pid]; // define particle
            n_particles = cell.np; // no. of particles in cell
            n_particles1 = cell.np1; // no. particles in cell partition 1
            n_particles2 = cell.np2;
    #ifdef PERIODIC
            particle.pos+=shift;
    #endif
            return 0;
        }

    public:
        int draw_particle_without_class(integer3 id_3D, Particle &particle, int &pid, Float3 shift, Grid *grid, int &n_particles, gsl_rng* locrng){
            // Draw a random particle from a cell given the cell ID.
            // This updates the particle and particle ID and returns 1 if error.
            // This is used for k,l cells (with no indication of particle random class)
            int id_1D = grid-> test_cell(id_3D);
            if(id_1D<0) return 1; // error if cell not in grid
            Cell cell = grid->c[id_1D];
            if(cell.np==0) return 1; // error if empty cell
            pid = floor(gsl_rng_uniform(locrng)*cell.np) + cell.start; // draw random ID
            particle = grid->p[pid]; // define particle
            n_particles = cell.np; // no. of particles in cell
    #ifdef PERIODIC
            particle.pos+=shift;
    #endif
            return 0;
        }
    public:
        void check_threads(Parameters *par,int print){
            // Set up OPENMP and define which threads to use
    #ifdef OPENMP
            cpu_set_t mask[par->nthread+1];
            int tnum=0;
            sched_getaffinity(0, sizeof(cpu_set_t), &mask[par->nthread]);
            if(print==1) fprintf(stderr, " CPUs used are: ");
            for(int ii=0;ii<64;ii++){
                if(CPU_ISSET(ii, &mask[par->nthread])){
                    if(print==1) fprintf(stderr,"%d ", ii);
                    CPU_ZERO(&mask[tnum]);
                    CPU_SET(ii,&mask[tnum]);
                    tnum++;
                }
            }
            fprintf(stderr,"\n");
    #endif
        }
    private:
        CorrelationFunction* which_cf(CorrelationFunction all_cf[], int Ia, int Ib){
            // Returns the relevant correlation function for two input field indices
            if((Ia==1)&(Ib==1)) return &all_cf[0];
            else if ((Ia==2)&&(Ib==2)) return &all_cf[1];
            else return &all_cf[2];
        }
        RandomDraws* which_rd(RandomDraws all_rd[], int Ia, int Ib){
            // Returns the relevant correlation function for two input field indices
            if((Ia==1)&(Ib==1)) return &all_rd[0];
            else if ((Ia==2)&&(Ib==2)) return &all_rd[1];
            else return &all_rd[2];
        }
        Grid* which_grid(Grid all_grid[], int Ia){
            // Returns the relevant correlation function for two input field indices
            if(Ia==1) return &all_grid[0];
            else return &all_grid[1];
        }

    public:
        compute_integral(){};

        compute_integral(Grid all_grid[], Parameters *par, CorrelationFunction all_cf[], RandomDraws all_rd[], int I1, int I2, int I3, int I4, int iter_no){
            // MAIN FUNCTION TO COMPUTE INTEGRALS

            int tot_iter=1; // total number of iterations
            if(par->multi_tracers==true) tot_iter=7;

            // Define relevant grids
            Grid *grid1 = which_grid(all_grid,I1);
            Grid *grid2 = which_grid(all_grid,I2);
            Grid *grid3 = which_grid(all_grid,I3);
            Grid *grid4 = which_grid(all_grid,I4);

            // Define relevant correlation functions
            CorrelationFunction *cf12 = which_cf(all_cf,I1,I2);
            CorrelationFunction *cf13 = which_cf(all_cf,I1,I3);
            CorrelationFunction *cf24 = which_cf(all_cf,I2,I4);

            // Define relevant random draw classes:
            RandomDraws *rd13 = which_rd(all_rd,I1,I3);
            RandomDraws *rd24 = which_rd(all_rd,I2,I4);

            nbin = par->nbin_short; // number of radial bins
            mbin = par->mbin; // number of mu bins

            STimer initial, TotalTime; // Time initialization
            initial.Start();

            int convergence_counter=0, printtime=0;// counter to stop loop early if convergence is reached.

    //-----------INITIALIZE OPENMP + CLASSES----------
            std::random_device urandom("/dev/urandom");
            std::uniform_int_distribution<unsigned int> dist(1, std::numeric_limits<unsigned int>::max());
            unsigned long int steps = dist(urandom);

            gsl_rng_env_setup(); // initialize gsl rng

            Integrals sumint(par, cf12, cf13, cf24, I1, I2, I3, I4); // total integral

            uint64 tot_quads=0; // global number of particle pairs/triples/quads used (including those rejected for being in the wrong bins)
            uint64 cell_attempt4=0; // number of j,k,l cells attempted
            uint64 used_cell4=0; // number of used j,k,l cells

            check_threads(par,1); // Define which threads we use

            initial.Stop();
            fprintf(stderr, "Init time: %g s\n",initial.Elapsed());
            printf("# 1st grid filled cells: %d\n",grid1->nf);
            printf("# All 1st grid points in use: %d\n",grid1->np);
            printf("# Max points in one cell in grid 1%d\n",grid1->maxnp);
            fflush(NULL);

            TotalTime.Start(); // Start timer

#ifdef OPENMP

    #pragma omp parallel firstprivate(steps,par,printtime,grid1,grid2,grid3,grid4,cf12,cf13,cf24) shared(sumint,TotalTime,gsl_rng_default,rd13,rd24) reduction(+:convergence_counter,cell_attempt4,used_cell4,tot_quads)
            { // start parallel loop
            // Decide which thread we are in
            int thread = omp_get_thread_num();
            assert(omp_get_num_threads()<=par->nthread);
            if (thread==0) printf("# Starting integral computation %d of %d on %d threads.\n", iter_no, tot_iter, omp_get_num_threads());
#else
            int thread = 0;
            printf("# Starting integral computation %d of %d single threaded.\n",iter_no,tot_iter);
            { // start loop
#endif

    //-----------DEFINE LOCAL THREAD VARIABLES
            Particle *prim_list; // list of particles in first cell
            int pln,sln,tln,fln,sln1,sln2; // number of particles in each cell
            int pid_j, pid_k, pid_l; // particle IDs particles drawn from j,k,l cell
            Particle particle_j, particle_k, particle_l; // randomly drawn particle
            //int* bin; // a-b bins for particles
            int* prim_ids; // list of particle IDs in primary cell
            double p2,p3,p4; // probabilities

            int *bin_ij; // i-j separation bin
            int mnp = grid1->maxnp; // max number of particles in a grid1 cell
            Float *xi_ik, *w_ijk, *w_ij; // arrays to store xi and weight values
            Float percent_counter;
            int x, prim_id_1D;
            integer3 delta2, delta3, delta4, prim_id, sec_id, thi_id;
            Float3 cell_sep2,cell_sep3;

            Integrals locint(par, cf12, cf13, cf24, I1, I2, I3, I4); // Accumulates the integral contribution of each thread

            gsl_rng* locrng = gsl_rng_alloc(gsl_rng_default); // one rng per thread
            gsl_rng_set(locrng, steps*(thread+1));

            // Assign memory for intermediate steps
            int ec=0;
            ec+=posix_memalign((void **) &prim_list, PAGE, sizeof(Particle)*mnp);
            ec+=posix_memalign((void **) &prim_ids, PAGE, sizeof(int)*mnp);
            ec+=posix_memalign((void **) &bin_ij, PAGE, sizeof(int)*mnp);
            ec+=posix_memalign((void **) &w_ij, PAGE, sizeof(Float)*mnp);
            ec+=posix_memalign((void **) &xi_ik, PAGE, sizeof(Float)*mnp);
            ec+=posix_memalign((void **) &w_ijk, PAGE, sizeof(Float)*mnp);
            assert(ec==0);

            uint64 loc_used_quads; // local counts of used pairs/triples/quads
    //-----------START FIRST LOOP-----------
    #ifdef OPENMP
    #pragma omp for schedule(dynamic)
    #endif
            for (int n_loops = 0; n_loops<par->max_loops; n_loops++){
                percent_counter=0.;
                loc_used_quads=0;

                // End loops early if convergence has been acheived
                if (convergence_counter==10){
                    if (printtime==0) printf("0.01%% convergence achieved in every bin 10 times, exiting.\n");
                    printtime++;
                    continue;
                    }
                // LOOP OVER ALL FILLED I CELLS
                for (int n1=0; n1<grid1->nf;n1++){

                    // Print time left
                    if((float(n1)/float(grid1->nf)*100)>=percent_counter){
                        printf("Integral %d of %d, run %d of %d on thread %d: Using cell %d of %d - %.0f percent complete\n",iter_no,tot_iter,1+n_loops/par->nthread, int(ceil(float(par->max_loops)/(float)par->nthread)),thread, n1+1,grid1->nf,percent_counter);
                        percent_counter+=5.;
                    }

                    // Pick first particle
                    prim_id_1D = grid1-> filled[n1]; // 1d ID for cell i
                    prim_id = grid1->cell_id_from_1d(prim_id_1D); // define first cell
                    pln = particle_list(prim_id_1D, prim_list, prim_ids, grid1); // update list of particles and number of particles

                    if(pln==0) continue; // skip if empty

                    loc_used_quads+=pln*par->N2*par->N3*par->N4;

                    // LOOP OVER N2 J CELLS
                    for (int n2=0; n2<par->N2; n2++){
                        // Draw second cell from i weighted by 1/r^2
                        delta2 = rd13->random_cubedraw_long(locrng, &p2); // can use any rd class here since drawing as 1/r^2
                        // p2 is the ratio of sampling to true pair distribution here
                        sec_id = prim_id + delta2;
                        cell_sep2 = grid2->cell_sep(delta2);
                        x = draw_particle(sec_id, particle_j, pid_j, cell_sep2, grid2, sln, locrng, sln1, sln2);
                        if(x==1) continue; // skip if error

                        // For all particles
                        p2*=1./(grid1->np*(double)sln); // probability is divided by total number of i particles and number of particles in cell

                        // LOOP OVER N3 K CELLS
                        for (int n3=0; n3<par->N3; n3++){
                            // Draw third cell from i weighted by xi(r)
                            delta3 = rd13->random_cubedraw(locrng, &p3); // use 1-3 random draw class here for xi_13
                            thi_id = prim_id + delta3;
                            cell_sep3 = grid3->cell_sep(delta3);
                            x = draw_particle_without_class(thi_id,particle_k,pid_k,cell_sep3,grid3,tln,locrng); // draw from third grid
                            if(x==1) continue;
                            if(pid_j==pid_k) continue;

                            p3*=p2/(double)tln; // update probability

                            // LOOP OVER N4 L CELLS
                            for (int n4=0; n4<par->N4; n4++){
                                cell_attempt4+=1; // new fourth cell attempted

                                // Draw fourth cell from j cell weighted by xi_24(r)
                                delta4 = rd24->random_cubedraw(locrng,&p4);
                                x = draw_particle_without_class(sec_id+delta4,particle_l,pid_l,cell_sep2+grid4->cell_sep(delta4),grid4,fln,locrng); // draw from 4th grid
                                if(x==1) continue;
                                if((pid_l==pid_j)||(pid_l==pid_k)) continue;

                                used_cell4+=1; // new fourth cell used

                                p4*=p3/(double)fln;


                                // Now compute the four-point integral
                                locint.fourth(prim_list, prim_ids, pln, particle_j, particle_k, particle_l, pid_j, pid_k, pid_l, p4);

                            }
                        }
                    }
                }

                // Update used pair/triple/quad counts
                tot_quads+=loc_used_quads;

    #ifdef OPENMP
    #pragma omp critical // only one processor can access at once
    #endif
            {
                if ((n_loops+1)%par->nthread==0){ // Print every nthread loops
                    TotalTime.Stop(); // interrupt timing to access .Elapsed()
                    int current_runtime = TotalTime.Elapsed();
                    int remaining_time = current_runtime/((n_loops+1)/par->nthread)*(par->max_loops/par->nthread-(n_loops+1)/par->nthread);  // estimated remaining time
                    fprintf(stderr,"\nFinished integral loop %d of %d after %d s. Estimated time left:  %2.2d:%2.2d:%2.2d hms, i.e. %d s.\n",n_loops+1,par->max_loops, current_runtime,remaining_time/3600,remaining_time/60%60, remaining_time%60,remaining_time);

                    TotalTime.Start(); // Restart the timer
                    Float rmsrd_C4, maxrd_C4;

                    sumint.rel_difference(&locint, rmsrd_C4, maxrd_C4);
                    if (maxrd_C4 < 1e-4) convergence_counter++;
                    if (n_loops!=0) {
                        fprintf(stderr, "RMS relative difference after loop %d is %.3f%%\n", n_loops, rmsrd_C4*100);
                        fprintf(stderr, "max relative difference after loop %d is %.3f%%\n", n_loops, maxrd_C4*100);
                    }
                }

                // Sum up integrals
                sumint.sum_ints(&locint);

                // Save output after each loop
                char output_string[50];
                sprintf(output_string,"%d", n_loops);

                locint.normalize();

                locint.save_integrals(output_string,1);

                locint.sum_total_counts(cnt4);
                locint.reset();
                }

            } // end cycle loop

            // Free up allocated memory at end of process
            free(prim_list);
            free(xi_ik);
            free(bin_ij);
            free(w_ij);
            free(w_ijk);
    } // end OPENMP loop

    //-----------REPORT + SAVE OUTPUT---------------
        TotalTime.Stop();

        // Normalize the accumulated results, using the RR counts
        sumint.normalize();

        int runtime = TotalTime.Elapsed();
        printf("\n\nINTEGRAL %d OF %d COMPLETE\n",iter_no,tot_iter);
        fprintf(stderr, "\nTotal process time for %.2e sets of cells and %.2e quads of particles: %d s, i.e. %2.2d:%2.2d:%2.2d hms\n", double(used_cell4),double(tot_quads),runtime, runtime/3600,runtime/60%60,runtime%60);
        printf("We tried %.2e quads of cells.\n",double(cell_attempt4));
        printf("Of these, we accepted %.2e quads of cells.\n",double(used_cell4));
        printf("We sampled %.2e quads of particles.\n",double(tot_quads));
        printf("Of these, we have integral contributions from %.2e quads of particles.\n",double(cnt4));
        printf("Cell acceptance ratio is %.3f for quads.\n",(double)used_cell4/cell_attempt4);

        printf("Acceptance ratio is %.3f for quads.\n",(double)cnt4/tot_quads);

        printf("\nTrial speed: %.2e quads per core per second\n",double(tot_quads)/(runtime*double(par->nthread)));
        printf("Acceptance speed: %.2e quads per core per second\n",double(cnt4)/(runtime*double(par->nthread)));

        char out_string[5];
        sprintf(out_string,"full");
        sumint.save_integrals(out_string,1); // save integrals to file
        sumint.save_counts(tot_quads); // save total pair/triple/quads attempted to file

        fflush(NULL);
        return;
        }

    };

#endif
