#pragma once
#ifndef CPPFETCH_UTILS_HPP
#define CPPFETCH_UTILS_HPP

#include <cstddef>
#include <filesystem>

namespace cppfetch {
    extern size_t write_callback(void* contents, size_t size, size_t nmemb, std::filesystem::path* file_path);
}  // namespace cppfetch

#endif