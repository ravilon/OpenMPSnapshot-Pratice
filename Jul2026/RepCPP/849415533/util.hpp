#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <filesystem>

std::string getDateTimeString(const bool withUnderscores, const int offsetSeconds);
std::ifstream loadTextFile(const std::filesystem::path& filePath);
time_t getCurrentTimeSeconds();
std::vector<std::string> splitStringByDelimiter(const std::string& string,
                                                const char delimiter);
std::vector<std::filesystem::directory_entry>
getFileEntries(const std::filesystem::path& dirPath);
void printProgress(const size_t progress, const size_t total);
std::ofstream createOutputFile(const std::filesystem::path& outputFilePath);
