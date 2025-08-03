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

#pragma once

#include <string>
#include <map>
#include <chrono>
#include <filesystem>
//
#include "libraries/nlohmann/json.hpp"
//
#include "plugin_constants.hpp"

enum class Config_Status {
    Success,
    Not_Found,
    Empty,
    Invalid
};

class Server_Info {
    public:
        std::string ip;
        std::string port;
        std::map<std::string, std::string> parameters;
        std::map<std::string, Config_Status> parameter_status;

        std::chrono::steady_clock::time_point last_update_time;
        std::chrono::steady_clock::time_point initial_success_time;
        std::map<std::string, std::string> connection_cache;
        bool connection_params_were_cached;

        bool initial_send_successful;
        bool environment_check_scheduled;
        bool environment_check_performed;
        
        std::filesystem::file_time_type hash_file_last_write_time;
        std::filesystem::file_time_type recovery_file_last_write_time;

        Server_Info() : last_update_time(std::chrono::steady_clock::now()),
            initial_success_time(),
            connection_params_were_cached(false),
            is_plugin_active(true),
            initial_send_successful(false),
            environment_check_scheduled(false),
            environment_check_performed(false) {
        }

        bool Initialize();
        bool Should_Send_Update() const;
        void Update_Last_Update_Time();

        bool Is_Plugin_Active() const {
            return is_plugin_active;
        }

        void Deactivate_Plugin() {
            is_plugin_active = false;
        }

        bool Reload_Parameters();
        void Cache_Connection_Parameters();
        void Restore_Connection_Parameters();
    private:
        bool Resolve_And_Verify_Configured_IP();
        bool Load_And_Validate_Parameters(bool is_reload = false);
        bool Validate_Section_Params(const nlohmann::ordered_json& section_json, const std::string& section_name, const Parameter_Config* param_configs, size_t param_count, bool& overall_config_ok, bool& needs_config_update_file);
        static bool Is_Valid_IP_Address_Or_Hostname(const std::string& input);

        bool is_plugin_active;
};