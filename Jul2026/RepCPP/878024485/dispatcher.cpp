#include "dependency_checker.h"
#include "pragma_injector.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <array>
#include <memory>
<<<<<<< HEAD
#include <iomanip>
=======
#include <cmath> // For std::pow
>>>>>>> 843f03bcccb86925fbd3f95638741f311d80856a

void log_execution(const std::string &mode, double executionTime, double calculatedSpeedup = -1.0, double actualSpeedup = -1.0) {
    std::ofstream logFile("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }
<<<<<<< HEAD

    // logFile << "======================== Execution Log ========================\n";
   
    logFile << "Run Mode       : " << (mode == "n" ? "Normal" : (mode == "c" ? "CPU" : "GPU")) << "\n";
=======
    logFile << "======================== Execution Log ========================\n";
    logFile << "Run Mode       : " << (mode) << "\n";
>>>>>>> 843f03bcccb86925fbd3f95638741f311d80856a
    logFile << "Execution Time : " << executionTime << " seconds\n";
    if (calculatedSpeedup != -1.0)
    {
       logFile << "Calculated Speedup : " << calculatedSpeedup << "\n";
    }
    if (actualSpeedup != -1.0)
    {
        logFile << "Actual Speedup     : " << actualSpeedup << "\n";
    }
    logFile.close();
}

// Function to estimate speedup using Amdahl's law
double estimate_amdahls_speedup(double p, int n) {
    if (p < 0 || p > 1) {
        std::cerr << "Error: p value should be between 0 and 1.\n";
       return -1.0;
    }

    if (n <=0 )
    {
       std::cerr << "Error: n value should be greater than 0.\n";
       return -1.0;
    }
    return 1.0 / ((1.0 - p) + (p / static_cast<double>(n)));
}

void log_gpu_stats() {
    const std::string logFolder = "logs/";
    const std::string gpuLogFilePath = logFolder + "gpu_stat.csv";

    // Ensure the logs directory exists
    system(("mkdir -p " + logFolder).c_str());

    std::ofstream gpuLogFile(gpuLogFilePath, std::ios::app);
    if (!gpuLogFile.is_open()) {
        std::cerr << "Error: Unable to write to GPU log file.\n";
        return;
    }

    // Log headers if the file is empty
    gpuLogFile.seekp(0, std::ios::end);
    if (gpuLogFile.tellp() == 0) {
        gpuLogFile << "Timestamp,Temperature (C),Power (W),Memory Used (MB),Utilization (%),Clock Speed (MHz)\n";
    }

    // Run nvidia-smi command and capture the output
    std::array<char, 256> buffer;
    std::string command = "nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits -i 0"; // GPU 0 stats
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    
    if (!pipe) {
        std::cerr << "Error: Unable to run nvidia-smi command.\n";
        return;
    }

    std::string result;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // Write the stats into the CSV file
    gpuLogFile << result;
    gpuLogFile.close();
}

void compile_and_execute(const std::string &filePath, char executionMode, double estimatedP = -1.0) {
     const std::string outputBinary = "build/main_parallel";
     const std::string compileCommand = (executionMode == 'n')
                                         ? "g++ -o " + outputBinary + " " + filePath
                                         : "g++ -fopenmp -o " + outputBinary + " " + filePath;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
<<<<<<< HEAD
    // log the start time
    std::ofstream logFile("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }
    logFile << "======================== Execution Log ========================\n";
    // logFile << "Start Time     : " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    auto startTimeFormatted = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logFile << "Start Time     : " << std::put_time(std::localtime(&startTimeFormatted), "%c %Z") << "\n";
    if (std::system(compileCommand.c_str()) != 0) {
        std::cerr << "Compilation failed!\n";
        return;
    }
    logFile.close();

    // Run the program
    std::string executeCommand = outputBinary;
    if (executionMode == 'g') {
        // Run GPU-specific actions
        std::thread gpuLogThread([&]() {
=======
     if (std::system(compileCommand.c_str()) != 0) {
            std::cerr << "Compilation failed!\n";
           return;
        }

        // Run the program
      std::string executeCommand = outputBinary;
        if (executionMode == 'g') {
          // Run GPU-specific actions
          std::thread gpuLogThread([&]() {
>>>>>>> 843f03bcccb86925fbd3f95638741f311d80856a
            while (true) {
              log_gpu_stats();
              std::this_thread::sleep_for(std::chrono::seconds(1)); // Log every second
            }
        });

        gpuLogThread.detach(); // Detach thread to continue logging GPU stats while running
      }

       auto executionStart = std::chrono::high_resolution_clock::now();
    if (std::system(executeCommand.c_str()) != 0) {
            std::cerr << "Execution failed!\n";
            return;
        }
    auto executionEnd = std::chrono::high_resolution_clock::now();
<<<<<<< HEAD
    // Calculate execution time
    std::chrono::duration<double> executionTime = executionEnd - executionStart;
    log_execution(executionMode == 'g' ? "GPU" : (executionMode == 'c' ? "CPU" : "Normal"), executionTime.count());

    // Stop logging GPU stats
    if (executionMode == 'g') {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait for the last log
        log_gpu_stats();  // Log one last time
    }

    // reopne the log file to log the end time
    logFile.open("logs/execution_log.txt", std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Unable to write to log file.\n";
        return;
    }
      
    // log the end time
    // logFile << "End Time       : " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    // format the end time
    auto endTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    logFile << "End Time       : " << std::put_time(std::localtime(&endTime), "%c %Z") << "\n";

    logFile.close();
=======

        // Calculate execution time
        std::chrono::duration<double> executionTime = executionEnd - executionStart;
      double actualExecutionTime = executionTime.count();

      double calculatedSpeedup = -1.0;

      if (executionMode != 'n' && estimatedP > 0.0) {
            int numberOfProcessors = (executionMode == 'c' ? 4 : 100);  // Example core and gpu numbers, you may need to change this
            calculatedSpeedup = estimate_amdahls_speedup(estimatedP, numberOfProcessors);
        }
      // Retrieve previous execution time from file
       double previousExecutionTime = -1.0;

       if(executionMode != 'n') {
          std::ifstream prevFile("logs/execution_log.txt");
          std::string line;
            while (std::getline(prevFile, line)) {
               if (line.find("Run Mode       : Normal") != std::string::npos)
                {
                    if(std::getline(prevFile, line)){
                         size_t pos = line.find("Execution Time : ");
                         if (pos != std::string::npos)
                         {
                            try{
                                previousExecutionTime = std::stod(line.substr(pos+17));
                            } catch(std::invalid_argument& e){
                                 previousExecutionTime = -1.0;
                            }
                         }
                       break;
                    }
                }
            }
            prevFile.close();
        }
        double actualSpeedup = -1.0;
         if (previousExecutionTime > 0 && executionMode != 'n')
        {
          actualSpeedup = previousExecutionTime / actualExecutionTime;
        }

        log_execution(executionMode == 'g' ? "GPU" : (executionMode == 'c' ? "CPU" : "Normal"), actualExecutionTime, calculatedSpeedup, actualSpeedup);
>>>>>>> 843f03bcccb86925fbd3f95638741f311d80856a
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
      std::cerr << "Usage: ./dispatcher <source_file> <mode: n|c|g|r>\n";
        return 1;
    }

    const std::string sourceFilePath = argv[1];
    char executionMode = argv[2][0];
    // std::cout << "Execution Mode is from dispatcher.cpp " << executionMode << "\n";

    if (executionMode == 'r') {
        // Randomly decide between CPU ('c') and GPU ('g')
        std::srand(std::time(nullptr));
        executionMode = (std::rand() % 2 == 0) ? 'c' : 'g';
    }

    if (executionMode == 'n') {
        std::cout << "Running program in Normal mode (no pragmas).\n";
        compile_and_execute(sourceFilePath, executionMode);
        return 0;
    }

    std::cout << "Checking dependencies...\n";
    if (!dependency_checker(sourceFilePath)) {
       std::cerr << "Error: Loop dependencies detected. Cannot parallelize.\n";
       return 1;
    }

    // Simplistic assumption that 70% of code is parallelizable
     double estimatedP = 0.7;

    std::cout << "Modifying source file for execution mode: "
                  << (executionMode == 'c' ? "CPU" : "GPU") << "\n";
    if (!add_pragma_to_loops(sourceFilePath, executionMode)) {
        std::cerr << "Error: Failed to add pragmas to the source file.\n";
        return 1;
     }

    compile_and_execute(sourceFilePath, executionMode, estimatedP);
    return 0;
}