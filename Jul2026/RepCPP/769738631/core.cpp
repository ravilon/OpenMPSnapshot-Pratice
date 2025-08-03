#include <curl/curl.h>
#include <omp.h>

#include <algorithm>
#include <cppfetch/core.hpp>
#include <cppfetch/utils.hpp>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace cppfetch {
/**
* @brief Default constructor for cppfetch class.
*/
cppfetch::cppfetch() : n_threads(omp_get_num_procs()), verbose(false) {}

/**
* @brief Get the list of files to be fetched.
*
* @return Reference to the vector containing the list of files.
*/
const std::vector<std::string>& cppfetch::get_files_list() const {
std::lock_guard<std::mutex> lock(mutex);
return this->files_list;
}

/**
* @brief Get the number of threads used for fetching files in parallel.
*
* @return Number of threads.
*/
int16_t cppfetch::get_n_threads() const {
std::lock_guard<std::mutex> lock(mutex);
return this->n_threads;
}

/**
* @brief Set verbose mode.
*
* @param flag Boolean flag indicating whether to enable verbose mode.
*/
void cppfetch::set_verbose(bool flag) {
std::lock_guard<std::mutex> lock(mutex);
this->verbose = flag;
}

/**
* @brief Add a file to the list of files to be fetched.
*
* @param file Path of the file to be added.
*/
void cppfetch::add_file(const std::string& file) {
std::lock_guard<std::mutex> lock(mutex);
this->files_list.push_back(file);
}

/**
* @brief Remove a file from the list of files to be fetched.
*
* @param file Path of the file to be removed.
*/
void cppfetch::remove_file(const std::string& file) {
std::lock_guard<std::mutex> lock(mutex);
auto it = std::find(this->files_list.begin(), this->files_list.end(), file);
if (it != this->files_list.end()) {
this->files_list.erase(it);
} else {
std::cerr << "File \"" << file << "\" has not been added to the list of files to be downloaded!\n";
}
}

/**
* @brief Download a single file from a given URL and save it to a specified path.
*
* @param file_url URL of the file to be downloaded.
* @param path_to_save Optional. Path where the file is saved. If not provided, the file is saved in the current
* directory with a default name produced starting from the URL name.
*/
void cppfetch::download_single_file(const std::string& file_url, const std::filesystem::path& path_to_save) const {
std::lock_guard<std::mutex> lock(mutex);
if (this->verbose) {
#pragma omp critical
{ std::cout << "Downloading file: " << file_url << "...\n"; }
}

curl_global_init(CURL_GLOBAL_ALL);
CURL* curl = curl_easy_init();

if (curl) {
curl_easy_setopt(curl, CURLOPT_URL, file_url.c_str());

std::string filename;
size_t last_slash = file_url.find_last_of('/');
if (last_slash != std::string::npos) {
filename = file_url.substr(last_slash + 1);
} else {
filename = "downloaded_file";
}
std::filesystem::path actual_path_to_save = path_to_save / filename;

if (std::filesystem::exists(actual_path_to_save) && std::filesystem::is_regular_file(actual_path_to_save)) {
std::cerr << "Error: a file named " << actual_path_to_save
<< " already exists! Operation is skipped!\n";
} else {
curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
curl_easy_setopt(curl, CURLOPT_WRITEDATA, &actual_path_to_save);

CURLcode res = curl_easy_perform(curl);
if (res != CURLE_OK) {
std::cerr << "Error while downloading: " << curl_easy_strerror(res) << "\n";
std::exit(0);
}
}
curl_easy_cleanup(curl);
} else {
std::cerr << "Cannot initialize libcurl!"
<< "\n";
std::exit(0);
}
}

/**
* @brief Check if verbose mode is enabled.
*
* @return True if verbose mode is enabled, false otherwise.
*/
bool cppfetch::is_verbose() const {
std::lock_guard<std::mutex> lock(mutex);
return this->verbose;
}

/**
* Download all files from the files list.
*
* @param path_to_save The path to save downloaded files.
* @param parallelize  Flag indicating whether to parallelize the download process.
*/
void cppfetch::download_all(const std::filesystem::path& path_to_save, bool parallelize) const {
#if defined(__clang__)
if (parallelize) {
#pragma omp parallel for
for (size_t i = 0; i < this->files_list.size(); ++i) {
this->download_single_file(this->files_list[i], path_to_save);
}
} else {
for (size_t i = 0; i < this->files_list.size(); ++i) {
this->download_single_file(this->files_list[i], path_to_save);
}
}
#else
if (parallelize) {
#pragma omp parallel for
for (const auto& file: this->files_list) {
this->download_single_file(file, path_to_save);
}
} else {
for (const auto& file: this->files_list) {
this->download_single_file(file, path_to_save);
}
}
#endif
}

/**
* Set the number of threads for parallel processing.
*
* @param n_threads_mod The number of threads to set.
*/
void cppfetch::set_n_threads(int16_t n_threads_mod) {
std::lock_guard<std::mutex> lock(mutex);
if (n_threads_mod > n_threads) {
std::cout << "Warning: the selected number of threads exceeds the maximum available threads!\n";
}
this->n_threads = n_threads_mod;
}
}  // namespace cppfetch
