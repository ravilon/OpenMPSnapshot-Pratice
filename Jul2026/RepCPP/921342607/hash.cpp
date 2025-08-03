/* ============================================================================ *
 * SPC Integration - Plugin for SA-MP (San Andreas Multiplayer)                 *
 * ============================================================================ *
 *                                                                              *
 * Copyright (c) 2025, SPC (SA-MP Programming Community)                        *
 * All rights reserved.                                                         *
 *                                                                              *
 * Developed by: Calasans                                                       *
 * Provided by: SA-MP Programming Community                                     *
 * Repository: https://github.com/spc-samp/spc-integration                      *
 *                                                                              *
 * ============================================================================ *
 *                                                                              *
 * Licensed under the Apache License, Version 2.0 (the "License");              *
 * you may not use this file except in compliance with the License.             *
 * You may obtain a copy of the License at:                                     *
 *                                                                              *
 *     http://www.apache.org/licenses/LICENSE-2.0                               *
 *                                                                              *
 * Unless required by applicable law or agreed to in writing, software          *
 * distributed under the License is distributed on an "AS IS" BASIS,            *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     *
 * See the License for the specific language governing permissions and          *
 * limitations under the License.                                               *
 *                                                                              *
 * ============================================================================ */

#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <cctype>
//
#if defined(_WIN32)
    #include <windows.h>
    #include <bcrypt.h>
    #include <memory>

    #pragma comment(lib, "bcrypt.lib")
#elif defined(__linux__)
    #include <openssl/sha.h>
    #include <openssl/rand.h>
    #include <openssl/err.h>
#endif
//
#include "hash.hpp"
#include "plugin_constants.hpp"
#include "logger.hpp"
#include "secret_manager.hpp"

namespace spc_sysf = std::filesystem;

namespace Hash {
    std::string Toggle_Hash(std::string_view data) {
        auto key_accessor = DRALYXOR_SECURE(Secret_Manager::Detail::Get_Hash_KEY());
        const char* key_ptr = key_accessor.Get();

        if (!key_ptr)
            return (Logger::Log_Formatted(Log_Level::ERROR_s, "Hash failed: Secret key decryption failed."), std::string(data));

        std::string key(key_ptr);

        if (key.empty())
            return (Logger::Log_Formatted(Log_Level::ERROR_s, "Hash failed: Secret key is empty."), std::string(data));

        std::string output(data);

        for (size_t i = 0; i < data.size(); ++i)
            output[i] = data[i] ^ key[i % key.length()];

        return output;
    }

    spc_sysf::path Get_Hash_File_Full_Path() {
        return spc_sysf::path(PLUGIN_DIRECTORY) / HASH_FILE;
    }

    std::string Generate_New_Hash() {
        unsigned char random_bytes[RANDOM_BYTES_COUNT];
        bool random_success = false;

#if defined(_WIN32)
        BCRYPT_ALG_HANDLE h_alg = nullptr;

        struct Alg_Handle_Closer {
            void operator()(BCRYPT_ALG_HANDLE h) {
                if (h)
                    BCryptCloseAlgorithmProvider(h, 0);
            }
        };
        std::unique_ptr<void, Alg_Handle_Closer> alg_handle_guard(nullptr);

        NTSTATUS status = BCryptOpenAlgorithmProvider(&h_alg, BCRYPT_RNG_ALGORITHM, MS_PRIMITIVE_PROVIDER, 0);

        if (BCRYPT_SUCCESS(status)) {
            alg_handle_guard.reset(h_alg);
            status = BCryptGenRandom(h_alg, random_bytes, RANDOM_BYTES_COUNT, 0);

            if (BCRYPT_SUCCESS(status))
                random_success = true;
            else
                Logger::Log(Log_Level::ERROR_s, "BCryptGenRandom failed with status 0x%x", status);
        }
        else
            Logger::Log(Log_Level::ERROR_s, "BCryptOpenAlgorithmProvider for RNG failed with status 0x%x", status);
#elif defined(__linux__)
        if (RAND_bytes(random_bytes, RANDOM_BYTES_COUNT) == 1)
            random_success = true;
        else
            Logger::Log(Log_Level::ERROR_s, "RAND_bytes failed. OpenSSL error: %s", ERR_error_string(ERR_get_error(), nullptr));
#endif
        if (!random_success)
            return (Logger::Log_Formatted(Log_Level::ERROR_s, "Secure random generation failed for hash. Aborting hash generation."), "");

        unsigned char hash_output[HASH_DIGEST_LENGTH];
#if defined(_WIN32)
        BCRYPT_ALG_HANDLE h_sha_alg = nullptr;
        BCRYPT_HASH_HANDLE h_hash = nullptr;

        struct Sha_Alg_Handle_Closer {
            void operator()(BCRYPT_ALG_HANDLE h) {
                if (h)
                    BCryptCloseAlgorithmProvider(h, 0);
            }
        };

        struct Sha_Hash_Handle_Closer {
            void operator()(BCRYPT_HASH_HANDLE h) {
                if (h)
                    BCryptDestroyHash(h);
            }
        };

        std::unique_ptr<void, Sha_Alg_Handle_Closer> sha_alg_guard(nullptr);
        std::unique_ptr<void, Sha_Hash_Handle_Closer> sha_hash_guard(nullptr);

        NTSTATUS sha_status = BCryptOpenAlgorithmProvider(&h_sha_alg, BCRYPT_SHA256_ALGORITHM, MS_PRIMITIVE_PROVIDER, 0);

        if (!BCRYPT_SUCCESS(sha_status))
            return (Logger::Log(Log_Level::ERROR_s, "BCryptOpenAlgorithmProvider for SHA256 failed: 0x%x", sha_status), "");

        sha_alg_guard.reset(h_sha_alg);
        sha_status = BCryptCreateHash(h_sha_alg, &h_hash, nullptr, 0, nullptr, 0, 0);

        if (!BCRYPT_SUCCESS(sha_status))
            return (Logger::Log(Log_Level::ERROR_s, "BCryptCreateHash for SHA256 failed: 0x%x", sha_status), "");

        sha_hash_guard.reset(h_hash);
        sha_status = BCryptHashData(h_hash, random_bytes, RANDOM_BYTES_COUNT, 0);

        if (!BCRYPT_SUCCESS(sha_status))
            return (Logger::Log(Log_Level::ERROR_s, "BCryptHashData for SHA256 failed: 0x%x", sha_status), "");

        sha_status = BCryptFinishHash(h_hash, hash_output, HASH_DIGEST_LENGTH, 0);

        if (!BCRYPT_SUCCESS(sha_status))
            return (Logger::Log(Log_Level::ERROR_s, "BCryptFinishHash for SHA256 failed: 0x%x", sha_status), "");
#elif defined(__linux__)
        SHA256_CTX sha256_ctx;

        if (!SHA256_Init(&sha256_ctx))
            return (Logger::Log(Log_Level::ERROR_s, "SHA256_Init failed. OpenSSL error: %s", ERR_error_string(ERR_get_error(), nullptr)), "");

        if (!SHA256_Update(&sha256_ctx, random_bytes, RANDOM_BYTES_COUNT))
            return (Logger::Log(Log_Level::ERROR_s, "SHA256_Update failed. OpenSSL error: %s", ERR_error_string(ERR_get_error(), nullptr)), "");

        if (!SHA256_Final(hash_output, &sha256_ctx))
            return (Logger::Log(Log_Level::ERROR_s, "SHA256_Final failed. OpenSSL error: %s", ERR_error_string(ERR_get_error(), nullptr)), "");
#endif
        std::stringstream ss_hex;
        ss_hex << std::hex << std::setfill('0');
        
        for (size_t i = 0; i < HASH_DIGEST_LENGTH; ++i)
            ss_hex << std::setw(2) << static_cast<unsigned int>(hash_output[i]);
        
        return ss_hex.str();
    }

    bool Save_Hash(const std::string& hash_to_save) {
        const spc_sysf::path hash_file_path = Get_Hash_File_Full_Path();
        const spc_sysf::path plugin_dir = hash_file_path.parent_path();

        try {
            if (!spc_sysf::exists(plugin_dir)) {
                if (!spc_sysf::create_directories(plugin_dir))
                    return (Logger::Log(Log_Level::ERROR_s, "Failed to create directory '%s' for hash file.", plugin_dir.string().c_str()), false);
            }
        }
        catch (const spc_sysf::filesystem_error& e) {
            return (Logger::Log(Log_Level::ERROR_s, "Filesystem error creating directory '%s': %s", plugin_dir.string().c_str(), e.what()), false);
        }

        std::ofstream hash_file_stream(hash_file_path, std::ios::out | std::ios::trunc | std::ios::binary);

        if (!hash_file_stream.is_open())
            return (Logger::Log(Log_Level::ERROR_s, "Failed to open hash file '%s' for writing.", hash_file_path.string().c_str()), false);

        std::string encrypted_hash = Toggle_Hash(hash_to_save);
        hash_file_stream << encrypted_hash;

        if (!hash_file_stream.good()) {
            Logger::Log(Log_Level::ERROR_s, "Failed to write all data to hash file '%s' (before close).", hash_file_path.string().c_str());
            hash_file_stream.close();

            return false;
        }

        hash_file_stream.close();

        if (!hash_file_stream.good())
            return (Logger::Log(Log_Level::ERROR_s, "Failed to write all data to hash file '%s' (after close/flush).", hash_file_path.string().c_str()), false);

        return true;
    }

    std::string Load_Hash() {
        const spc_sysf::path hash_file_path = Get_Hash_File_Full_Path();

        if (!spc_sysf::exists(hash_file_path))
            return "";
        
        std::ifstream hash_file_stream(hash_file_path, std::ios::in | std::ios::binary);

        if (!hash_file_stream.is_open())
            return (Logger::Log(Log_Level::ERROR_s, "Failed to open hash file '%s' for reading.", hash_file_path.string().c_str()), "");

        std::string encrypted_content((std::istreambuf_iterator<char>(hash_file_stream)), std::istreambuf_iterator<char>());
        hash_file_stream.close();

        if (encrypted_content.empty())
            return (Logger::Log(Log_Level::WARNING, "Loaded hash file '%s' is empty.", hash_file_path.string().c_str()), "");

        std::string decrypted_hash = Toggle_Hash(encrypted_content);

        if (decrypted_hash.length() != 64 || !std::all_of(decrypted_hash.begin(), decrypted_hash.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); }))
            return (Logger::Log(Log_Level::WARNING, "Decrypted hash from '%s' has invalid format or length. Potential wrong key or file corruption.", hash_file_path.string().c_str()), "");

        return decrypted_hash;
    }

    bool Delete_Hash_File() {
        const spc_sysf::path hash_file_path = Get_Hash_File_Full_Path();
        
        if (spc_sysf::exists(hash_file_path)) {
            Logger::Log(Log_Level::INFO, "Attempting to delete local server hash file '%s'...", hash_file_path.string().c_str());

            std::error_code ec;

            if (!spc_sysf::remove(hash_file_path, ec))
                return (Logger::Log(Log_Level::WARNING, "Failed to delete hash file '%s'. Error code: %d, Message: %s", hash_file_path.string().c_str(), ec.value(), ec.message().c_str()), false);

            Logger::Log_Formatted(Log_Level::INFO, "Hash file deleted successfully.");

            return true;
        }
        
        Logger::Log_Formatted(Log_Level::INFO, "Skipping hash file deletion: File does not exist.");

        return true;
    }

    bool Verify_Hash_File(std::string_view expected_hash) {
        std::string loaded_hash_val = Load_Hash();

        if (loaded_hash_val.empty())
            return false;

        if (loaded_hash_val != expected_hash) {
            std::string truncated_expected = expected_hash.length() > 12 ? std::string(expected_hash.substr(0, 12)) + "..." : std::string(expected_hash);
            std::string truncated_found = loaded_hash_val.length() > 12 ? loaded_hash_val.substr(0, 12) + "..." : loaded_hash_val;
            const spc_sysf::path hash_file_path = Get_Hash_File_Full_Path();

            Logger::Log(Log_Level::WARNING, "Hash mismatch in '%s'. Expected: '%s', Found: '%s'. Potential tampering.", hash_file_path.string().c_str(), truncated_expected.c_str(), truncated_found.c_str());

            return false;
        }
        
        return true;
    }
}