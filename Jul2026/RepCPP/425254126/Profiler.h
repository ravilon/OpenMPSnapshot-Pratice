#pragma once

#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

// Allows you to declare a unique identifier within a given cpp file
#define UNIQ_ID(line) UNIQ_ID_IMPL(line)
#define UNIQ_ID_IMPL(line) loc_var_##line
#define LOG_DURATION(message) LogDuration UNIQ_ID(__LINE__){message};

class LogDuration {
private:
    // The moment of the beginning of the measurement of the program execution time
    steady_clock::time_point start;
    string message;

public:
    // Adding explicit - protection against implicit conversion string to LogDuration
    explicit LogDuration(const string& msg = "") : start(steady_clock::now()), message(msg + ": ")
    {
    }

    ~LogDuration() {
        auto finish = steady_clock::now();
        std::cout << message << duration_cast<milliseconds>(finish - start).count() << " ms ("
                  << duration_cast<seconds>(finish - start).count() << " s)" << endl;
    }

};