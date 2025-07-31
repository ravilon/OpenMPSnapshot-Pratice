// #include "pragma_injector.h"
// #include <iostream>
// #include <fstream>
// #include <regex>

// bool add_pragma_to_loops(const std::string &filePath, char executionMode) {
//     std::ifstream inputFile(filePath);
//     if (!inputFile.is_open()) {
//         std::cerr << "Error: Unable to open source file.\n";
//         return false;
//     }

//     std::string modifiedCode;
//     std::string line;
//     std::regex loopRegex(R"((for|while)\s*\(.*\))");
//     bool pragmaAdded = false;

//     while (std::getline(inputFile, line)) {
//         if (std::regex_search(line, loopRegex) && !pragmaAdded) {
//             if (executionMode == 'c') {
//                 modifiedCode += "#pragma omp parallel for\n";
//             } else if (executionMode == 'g') {
//                 modifiedCode += "#pragma omp target teams distribute parallel for\n";
//             }
//             pragmaAdded = true; // Add pragma only to the first loop
//         }
//         modifiedCode += line + "\n";
//     }
//     inputFile.close();

//     // Write back the modified code
//     std::ofstream outputFile(filePath);
//     if (!outputFile.is_open()) {
//         std::cerr << "Error: Unable to write to source file.\n";
//         return false;
//     }
//     outputFile << modifiedCode;
//     outputFile.close();
//     return true;
// }

#include "pragma_injector.h"
#include <iostream>
#include <fstream>
#include <regex>
#include <stack>

bool add_pragma_to_loops(const std::string &filePath, char executionMode) {
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open source file.\n";
        return false;
    }

    std::string modifiedCode;
    std::string line;
    std::regex loopRegex(R"((for|while)\s*\(.*\))"); // Correct regex with escaped brace
    std::stack<int> braceBalance;

    bool inLoop = false;
    bool outerLoop = true;

    while (std::getline(inputFile, line)) {
        //Check for loop start - improved regex to handle various loop conditions
        if (std::regex_search(line, loopRegex) && outerLoop) {
            // std::cout << "Found Outer Loop: " << line << "\n";
            outerLoop = false;
            inLoop = true;
            if (executionMode == 'c') {
                modifiedCode += "#pragma omp parallel for\n";
            } else if (executionMode == 'g') {
                modifiedCode += "#pragma omp target teams distribute parallel for\n";
            }
            // braceBalance.push(1);
        }

        // Count braces to handle nested loops
        if(inLoop){
            for (char c : line) {
            if (c == '{') {
                // std::cout << "Found bracket { in line: " << line << "\n";
                braceBalance.push(1);
            } else if (c == '}') {
                // std::cout << "Found bracket } in line: " << line << "\n";
                if (!braceBalance.empty()) {
                    braceBalance.pop();
                    if (braceBalance.empty()) {
                        inLoop = false;
                        outerLoop = true;
                    }
                }
            }
        }
        }
        
        modifiedCode += line + "\n";
    }
    inputFile.close();

    std::ofstream outputFile(filePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to write to source file.\n";
        return false;
    }
    outputFile << modifiedCode;
    outputFile.close();
    return true;
}