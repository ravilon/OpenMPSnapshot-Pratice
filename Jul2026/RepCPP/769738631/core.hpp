#pragma once
#ifndef CPPFETCH_CORE_HPP
#define CPPFETCH_CORE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace cppfetch {
    class cppfetch {
        public:

            // Constructors
            explicit cppfetch();

            // Getters
            const std::vector<std::string>& get_files_list() const;
            int16_t get_n_threads() const;

            // Setters
            void set_verbose(bool flag);
            void set_n_threads(int16_t n_threads_mod);

            // Methods
            void download_single_file(const std::string& file_url,
                                      const std::filesystem::path& path_to_save = "") const;
            void add_file(const std::string& file);
            bool is_verbose() const;
            void remove_file(const std::string& file);
            void download_all(const std::filesystem::path& path_to_save = "", bool parallelize = true) const;

        private:

            // Members
            std::vector<std::string> files_list;
            int16_t n_threads;
            bool verbose;
            mutable std::mutex mutex;
    };
}  // namespace cppfetch

#endif
