/*
 * Created by Victor Youdom Kemmoe
 */

#include "n_body.h"

n_body::n_body(const std::string& file_path_, double delta_t_, int T_, int is_each_step_, int number_of_threads_, int process_id_, int process_master_, int comm_sz_) {
	/*
	 * assign values to different variables and
	 * load data before performing any operations
	 */

	file_path = file_path_;
	delta_t = delta_t_;
	T = T_;
	is_each_step = is_each_step_;
	number_of_threads = number_of_threads_;
	process_id = process_id_;
	process_master = process_master_;
	comm_sz = comm_sz_;

//read data from file
	read_data_from_txt();
}

void n_body::initialize_arrays() {
	/*
	 * initialize the size of the different array used by the program
	 */
	 local_n = n/comm_sz;

	 masses = (double *)malloc(n* sizeof(double));
	 velocities = (double *)malloc(n* 2 * sizeof(double));
	 positions = (double *)malloc(n* 2 * sizeof(double));

	 local_velocities = (double *)malloc(local_n* 2 * sizeof(double));
	 local_positions = (double *)malloc(local_n* 2 * sizeof(double));
	 local_forces = (double *)malloc(local_n * 2 * sizeof(double));

	 memset(local_forces, 0, local_n * 2 * sizeof(double));
}

void n_body::actualize_speed_pos() {



		for(int timestep=0; timestep<T; timestep++){
			if(is_each_step == 1){
				if(process_id==process_master){
				//process master == process 0 write to the file
					write_data_to_txt(timestep*delta_t);
				}
			}
#pragma omp parallel for num_threads(number_of_threads)
			for(int q=0; q<local_n; q++){
				for(int k = 0; k<n ;k++){
					//transform local_q into global q in to avoid misinterpretion and keep the basis used in serial basic implementation
					if(k!=(q + process_id*local_n )){

						double x_diff = *(positions + process_id*local_n*2 + q*2 + 0) - *(positions + k*2 + 0);
						double y_diff = *(positions + process_id*local_n*2+ q*2 + 1) - *(positions + k*2 + 1);

						double dist = sqrt(x_diff*x_diff + y_diff*y_diff);

						double dist_cubed = dist*dist*dist;
						*(local_forces + q*2 + 0) -= G*masses[process_id*local_n + q]*masses[k]/dist_cubed * x_diff;
						*(local_forces + q*2 + 1) -= G*masses[process_id*local_n + q]*masses[k]/dist_cubed * y_diff;

					}
				}
			}

#pragma omp parallel for num_threads(number_of_threads)
			for(int q = 0; q < local_n ; ++q){
				//updating velocity and speed
				*(local_positions + q*2 + 0) += delta_t * (*(local_velocities + q*2 + 0));
				*(local_positions + q*2 + 1) += delta_t * (*(local_velocities + q*2 + 1));
				*(local_velocities + q*2 + 0) += delta_t/masses[process_id*local_n + q] * (*(local_forces + q*2 + 0));
				*(local_velocities + q*2 + 1) += delta_t/masses[process_id*local_n + q] * (*(local_forces + q*2 + 1));
			}

			//Algather positions and velocities to actualise their values on all process and also
			//to pint out update value by process 0
			MPI_Allgather(local_positions, local_n*2, MPI_DOUBLE, positions, local_n*2, MPI_DOUBLE, MPI_COMM_WORLD);
			MPI_Allgather(local_velocities,local_n*2, MPI_DOUBLE, velocities,local_n*2, MPI_DOUBLE, MPI_COMM_WORLD);
		}

	if(process_id==process_master){
		write_data_to_txt(T*delta_t);
	}
}

void n_body::write_data_to_txt(double timestep) {
	/*
	 * Write calculated data to a file.
	 * create the file that's going to be used if not already present
	 */
    std::string outfile_name = std::string("data_")+ std::to_string(timestep);
    std::ofstream outfile (outfile_name);
    outfile<<n<<std::endl;
    for (int q = 0; q < n ; ++q) {
        outfile<<*(positions +q*2 +0)<<","<<*(positions +q*2 +1)<<","<<*(velocities +q*2 +0)<<","<<*(velocities +q*2 +1)<<","<<masses[q]<<std::endl;
    }
    outfile.close();
}

void n_body::read_data_from_txt() {
    /* open file containing data and read them
     * format of the file is:
     * first line represent number of data
     * rest of file is: x-position, y-position, x-velocity, y-velocity, masses
     *initialize arrays after reading the first line and getting the number of data at line == 1
     */
	int was_read = 0;

	if(process_id == process_master){
		//process zero is reading
		std::ifstream file(file_path.c_str());
		if(file.is_open()) {
			was_read = 1; //was_read OK
			std::string line;
			getline(file, line); //read first line

			n = std::stoi(line); //get number of elements from first line

			initialize_arrays(); //initialize the size of the different arrays that are going to hold the different values

			int i = 0;
			while (getline(file, line)) {
				std::stringstream linestream(line);
				std::string value;

				getline(linestream, value, ',');
				*(positions + i*2 + 0) = std::stod(value); // X position

				getline(linestream, value, ',');
				*(positions + i*2 + 1) = std::stod(value); // Y position

				getline(linestream, value, ',');
				*(velocities + i*2 + 0) = std::stod(value); // X velocity

				getline(linestream, value, ',');
				*(velocities + i*2 + 1) = std::stod(value); // Y velocity

				getline(linestream, value, ',');
				masses[i] = std::stod(value);

				i++;
			}

			file.close();
		}
	}

	MPI_Bcast(&was_read,1, MPI_INT, process_master, MPI_COMM_WORLD);

	if(was_read==0){
		//kill the program if process master was not able to read the file
		std::cerr<<"Error while opening the file"<<std::endl;
		exit(EXIT_FAILURE);
	}
	else{
		//read was ok, other process can initialise their array so that they
		//can receive data

		MPI_Bcast(&n,1, MPI_INT, process_master, MPI_COMM_WORLD);
		//Cast the value of n to every one so that remaining process can initialize their array
		if(process_id!=process_master){
			//all other process except process master initialize their arrays
			initialize_arrays();
		}

		//share data among process
		MPI_Bcast(positions,n* 2, MPI_DOUBLE, process_master, MPI_COMM_WORLD);
		MPI_Bcast(masses,n, MPI_DOUBLE, process_master, MPI_COMM_WORLD);
		MPI_Scatter(velocities, local_n* 2, MPI_DOUBLE, local_velocities, local_n* 2, MPI_DOUBLE, process_master,MPI_COMM_WORLD);

		//scatter position in order to initialize local position
		MPI_Scatter(positions, local_n* 2, MPI_DOUBLE, local_positions, local_n* 2, MPI_DOUBLE, process_master,MPI_COMM_WORLD);


	}
}
