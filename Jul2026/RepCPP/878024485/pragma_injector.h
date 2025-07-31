#ifndef PRAGMA_INJECTOR_H
#define PRAGMA_INJECTOR_H

#include <string>

// Function to add OpenMP pragma directives to loops in the source file
bool add_pragma_to_loops(const std::string &filePath, char executionMode);

#endif // PRAGMA_INJECTOR_H

