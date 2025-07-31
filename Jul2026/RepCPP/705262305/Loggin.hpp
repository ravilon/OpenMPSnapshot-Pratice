#pragma once

#include <cstdlib>
#include <iostream>

namespace logging {

// ANSI escape codes for text colors
const char *RESET = "\033[0m";
const char *BLACK = "\033[30m";
const char *RED = "\033[31m";
const char *GREEN = "\033[32m";
const char *YELLOW = "\033[33m";
const char *BLUE = "\033[34m";
const char *MAGENTA = "\033[35m";
const char *CYAN = "\033[36m";
const char *WHITE = "\033[37m";
const char *BOLDBLACK = "\033[1m\033[30m";
const char *BOLDRED = "\033[1m\033[31m";
const char *BOLDGREEN = "\033[1m\033[32m";
const char *BOLDYELLOW = "\033[1m\033[33m";
const char *BOLDBLUE = "\033[1m\033[34m";
const char *BOLDMAGENTA = "\033[1m\033[35m";
const char *BOLDCYAN = "\033[1m\033[36m";
const char *BOLDWHITE = "\033[1m\033[37m";

// Function to print colored message
void print_colored_message(const char *prefix, const char *msg,
                           const char *color) {
  std::cerr << color << "[" << prefix << "]:" << RESET << " " << msg << "\n";
}

// macros to print messages with colors
#define INFO(msg) print_colored_message("INFO", msg, logging::BOLDGREEN)

#define WARNING(msg) print_colored_message("WARNING", msg, logging::BOLDYELLOW)

#define ERROR(msg) print_colored_message("ERROR", msg, logging::BOLDRED);

} // namespace logging
