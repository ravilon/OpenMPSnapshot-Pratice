/*
 * This application is used to test the pinning of a process to a specific core.
 * The application uses MPI to create processes that report on which node they are running.
 * Each process uses OpenMP to create threads that report on which core they are running.
 * Each thread also performs a busy wait to keep the core busy, so that it can be monitored whether the thread remains on the same core.
 *
 * The application takes the following arguments:
 *   - duration: the duration of the busy wait in seconds, default value 1,
 *     the duration should be a positive integer
 *   - cycles: the number of busy waits to perform, default value 2,
 *     the cycles should be a positive integer, or zero
 *   - version: print the version number
 *   - help: print the help message
 *
 * The application uses Boost to parse the command line arguments.
 *
 * Optionally, the application can be build without MPI to cover cases when MPI is not available.
 * This is controlled by the WITH_MPI macro.
 */

#include <boost/asio/ip/host_name.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include <sched.h>
#include <sstream>

// Define the version number to silence compiler errors while editing
#ifndef PINTEST_VERSION
#define PINTEST_VERSION "not set"
#endif

// Structure to hold the command line arguments
struct Arguments {
    int duration;
    int cycles;
};

// Enumeration to hold parse_arguments status
enum class ParseStatus {
    OK,
    HELP,
    VERSION,
    ERROR
};

// Parse the command line arguments
std::tuple<Arguments, ParseStatus> parse_arguments(int argc, char* argv[]) {
    Arguments args;
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("duration", po::value<int>(&args.duration)->default_value(1),
         "duration of the busy wait in seconds")
        ("cycles", po::value<int>(&args.cycles)->default_value(2),
         "number of busy waits to perform")
        ("version", "print version number");

    ParseStatus status {ParseStatus::OK};

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        status = ParseStatus::ERROR;
        return std::make_tuple(args, status);
    }
    po::notify(vm);

    // check if help is requested
    if (vm.count("help") || !vm.count("duration") || !vm.count("cycles")) {
        std::cout << desc << std::endl;
        status = ParseStatus::HELP;
    }

    // check if version is requested
    if (vm.count("version")) {
        std::cout << "PinTest version " << PINTEST_VERSION << std::endl;    
        status = ParseStatus::VERSION;
    }

    // check if the duration and cycles are valid
    if (args.duration <= 0) {
        std::cerr << "Error: duration should be a positive integer" << std::endl;
        status = ParseStatus::ERROR;
    }
    if (args.cycles < 0) {
        std::cerr << "Error: cycles should be a positive integer, or zero" << std::endl;
        status = ParseStatus::ERROR;
    }

    return std::make_tuple(args, status);
}
//
// function to create and commit a custom MPI data type for the Arguments structure
#ifdef WITH_MPI
void create_mpi_arguments_type(MPI_Datatype* arguments_type) {
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    int block_lengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(Arguments, duration);
    offsets[1] = offsetof(Arguments, cycles);
    MPI_Type_create_struct(2, block_lengths, offsets, types, arguments_type);
    MPI_Type_commit(arguments_type);
}
#endif

// function that lets rank 0 parse the command line arguments and broadcast them to all other ranks
Arguments broadcast_arguments(int argc, char* argv[]) {
    int rank {0};
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype arguments_type;
    create_mpi_arguments_type(&arguments_type);
#endif
    Arguments args {};
    ParseStatus status {};
    if (rank == 0) {
        std::tie(args, status) = parse_arguments(argc, argv);
    }

    // If an error occurred, abort the program
    if (status == ParseStatus::ERROR) {
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);   
#else
        std::exit(1);
#endif
    }

    // If help or version is requested, finalize MPI and exit
    if (status == ParseStatus::HELP || status == ParseStatus::VERSION) {
#ifdef WITH_MPI
        MPI_Finalize();
#endif
        std::exit(0);
    }

#ifdef WITH_MPI
    MPI_Bcast(&args, 1, arguments_type, 0, MPI_COMM_WORLD);

#endif
    return args;
}

// function that provides runtime feedback on the command line arguments,
// the number of ranks, and the number of threads
void report_runtime_parameters(const Arguments& args) {
    int rank {0};
    int size {1};
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    if (rank == 0) {
        // determine the number of threads
        int threads {1};
#ifdef _OPENMP
#pragma omp parallel
        threads = omp_get_max_threads();
#endif
        std::stringstream msg;
        msg << "# Running with\n"
            << "#     ranks=" << size << "\n"
            << "#     threads=" << threads << "\n"
            << "#     duration=" << args.duration << "\n"
            << "#     cycles=" << args.cycles << std::endl;
        std::cout << msg.str();
    }
}

// function to create a timestamp in the format "YYYY-MM-DD HH:MM:SS.mmm"
std::string create_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_c);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    char buffer[80];
    std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", &now_tm);
    return std::string(buffer) + "." + std::to_string(now_ms.count());
}

// Report on which node the process is running
void report_node() {
    int rank {0};
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    const auto hostname = boost::asio::ip::host_name();
    std::stringstream msg;
    msg << "[" << create_timestamp() << "] "
        << "Rank=" << rank << ", node=" << hostname << "\n";
    std::cout << msg.str();
}

// Report on which core the thread is running
void report_core() {
    int rank {0};
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    int thread_id {0};
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    const auto core_id = sched_getcpu();
    std::stringstream msg;
    msg << "[" << create_timestamp() << "] ";
    msg << "Rank=" << rank
        << ", Thread=" << thread_id
        << ", core=" << core_id << "\n";
    std::cout << msg.str();
}

// Perform a busy wait
void busy_wait(int duration) {
    auto start = std::chrono::high_resolution_clock::now();
    for (;;) {
        auto now {std::chrono::high_resolution_clock::now()};
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= duration) {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
#ifdef WITH_MPI
    MPI_Init(&argc, &argv);
#endif
    auto args = broadcast_arguments(argc, argv);
    report_runtime_parameters(args);
#ifdef WITH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    report_node();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // Report initial core binding
        report_core();
        for (int i = 0; i < args.cycles; ++i) {
            busy_wait(args.duration);
            // Report core binding after busy wait to check pinning over time
            report_core();
        }
    }
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
