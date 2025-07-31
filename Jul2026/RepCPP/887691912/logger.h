#pragma once

#include <chrono>
#include <string>
#include <vector>

using chrono_clock = std::chrono::high_resolution_clock;

class Logger {
   public:
    Logger(const std::string& filename, int num_steps);

    void log_start();
    void log_finish();

    void log_iteration(double best_fitness, double avg_fitness, double med_fitness,
                       int best_score, double avg_score, int med_score, double step_time);

    void print_iteration_summary();
    void export_log();

   private:
    std::string filename;

    std::vector<double> best_fitness, avg_fitness, med_fitness;
    std::vector<int> best_score;
    std::vector<double> avg_score;
    std::vector<int> med_score;
    std::vector<double> step_times;

    chrono_clock::time_point start_time;
    std::chrono::duration<double> total_time;
};