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

#include <cstddef>
#include <chrono>

// Plugin Information
constexpr char PLUGIN_NAME[] = "SPC Integration";
constexpr char PLUGIN_DEVELOPER[] = "Calasans";
constexpr char PLUGIN_VERSION[] = "1.0.1";
constexpr char PLUGIN_USER_AGENT_BASE[] = "SPC-Integration-Plugin/";

// Directory and File Configuration
constexpr char PLUGIN_DIRECTORY[] = "spc-integration";
constexpr char CONFIG_FILE[] = "config.json";
constexpr char LOG_FILE[] = "logs.log";
constexpr char HASH_FILE[] = "hash.hash";
constexpr char RESERVED_FILE[] = "reserved.recovery";

// Configuration JSON Section & Key Names
constexpr char CONFIG_SECTION_CONNECTION[] = "connection";
constexpr char CONFIG_SECTION_PUBLIC_INFO[] = "public_information";
constexpr char JSON_KEY_IP[] = "ip";
constexpr char JSON_KEY_PORT[] = "port";
constexpr char JSON_KEY_PRODUCTION[] = "production_environment";
constexpr char JSON_KEY_TEAM[] = "team";
constexpr char JSON_KEY_STATUS[] = "status";
constexpr char JSON_KEY_ERROR[] = "error";

// Security & Hashing
constexpr size_t HASH_DIGEST_LENGTH = 32;
constexpr size_t RANDOM_BYTES_COUNT = 32;

// Backend Actions & Parameter Keys
constexpr char ACTION_CREATE[] = "create";
constexpr char ACTION_UPDATE[] = "update";
constexpr char ACTION_DELETE[] = "delete";
constexpr char ACTION_CONFIRM_HASH[] = "confirm_hash";

constexpr char BACKEND_KEY_SERVER_IP[] = "server_ip";
constexpr char BACKEND_KEY_SERVER_PORT[] = "server_port";
constexpr char BACKEND_KEY_SERVER_HASH[] = "server_hash";
constexpr char BACKEND_SUFFIX_ENCODED[] = "_encoded";
constexpr char BACKEND_SUFFIX_URL[] = "_url";

// Plugin Behavior & Timings
constexpr auto UPDATE_INTERVAL = std::chrono::minutes(5);
constexpr int OFFLINE_THRESHOLD_MINUTES = 10;
constexpr auto ENVIRONMENT_CHECK_DELAY = std::chrono::minutes(7);
constexpr auto URL_VALIDATION_TIMEOUT = std::chrono::seconds(10);
constexpr int SHUTDOWN_GRACE_PERIOD_MS = 1500;

// Networking & Buffers
constexpr int DEFAULT_API_TIMEOUT_MS = 60000;
constexpr int ENV_CHECK_TIMEOUT_MS = 15000;
constexpr int IP_SERVICE_REQUEST_TIMEOUT_MS = 7000;
constexpr int IP_SERVICE_FETCH_TIMEOUT_SECONDS = 8;
constexpr size_t HTTP_RESPONSE_BUFFER_SIZE = 8192;
constexpr int MAX_RETRIES = 5;
constexpr auto RETRY_DELAY = std::chrono::seconds(2);

// Logging & Display
constexpr int BANNER_LINE_WIDTH = 60;
constexpr int MAX_LOG_BODY_PREVIEW_LEN = 512;
constexpr int IP_SERVICE_LOG_PREVIEW_LEN = 64;
constexpr int URL_LOG_PREVIEW_LEN = 70;
constexpr int HASH_TRUNCATE_LEN = 12;
constexpr int PUBLIC_INFO_TRUNCATE_LEN = 60;
constexpr int TEAM_MEMBER_JSON_TRUNCATE_LEN = 30;

// URL Validation & Special Values
constexpr size_t MAX_HEADER_READ_SIZE = 64;
constexpr int MAX_URL_LENGTH = 2048;

constexpr char URL_NOT_DEFINED[] = "not-defined";
constexpr char URL_STATUS_ONLINE[] = "online";

constexpr char URL_VAL_TYPE_GENERIC[] = "generic";
constexpr char URL_VAL_TYPE_IMAGE[] = "image";
constexpr char URL_VAL_TYPE_DISCORD[] = "discord";
constexpr char URL_VAL_TYPE_YOUTUBE[] = "youtube";
constexpr char URL_VAL_TYPE_INSTAGRAM[] = "instagram";
constexpr char URL_VAL_TYPE_FACEBOOK[] = "facebook";
constexpr char URL_VAL_TYPE_TIKTOK[] = "tiktok";

constexpr char LOCALHOST_IP[] = "127.0.0.1";
constexpr char LOCALHOST_NAME[] = "localhost";

// Parameter Configuration
struct Parameter_Config {
    const char* name;
    const char* default_value;
    bool required;
    bool base64_encoded;
    bool url_validated;
    const char* url_validation_type;
    bool send_to_backend;
    bool is_integer;
    bool log_output;
};

constexpr Parameter_Config CONNECTION_PARAMETERS[] = {
    {JSON_KEY_IP, LOCALHOST_IP, true, false, false, "", false, false, false},
    {JSON_KEY_PORT, "7777", true, false, false, "", false, true, false},
    {JSON_KEY_PRODUCTION, "true", true, false, false, "", false, false, false}
};
constexpr size_t CONNECTION_PARAMETER_COUNT = sizeof(CONNECTION_PARAMETERS) / sizeof(CONNECTION_PARAMETERS[0]);

constexpr Parameter_Config PUBLIC_INFORMATION_PARAMETERS[] = {
    {"logo", "https://example.com/image.png", true, true, true, URL_VAL_TYPE_IMAGE, true, false, true},
    {"banner", "https://example.com/banner.png", true, true, true, URL_VAL_TYPE_IMAGE, true, false, true},
    {"website", "https://example.com", true, true, true, URL_VAL_TYPE_GENERIC, true, false, true},
    {"discord", URL_NOT_DEFINED, false, true, true, URL_VAL_TYPE_DISCORD, true, false, true},
    {"youtube", URL_NOT_DEFINED, false, true, true, URL_VAL_TYPE_YOUTUBE, true, false, true},
    {"instagram", URL_NOT_DEFINED, false, true, true, URL_VAL_TYPE_INSTAGRAM, true, false, true},
    {"facebook", URL_NOT_DEFINED, false, true, true, URL_VAL_TYPE_FACEBOOK, true, false, true},
    {"tiktok", URL_NOT_DEFINED, false, true, true, URL_VAL_TYPE_TIKTOK, true, false, true},
    {JSON_KEY_TEAM, "{\"team\": [{\"name\": \"Your Name\", \"function\": \"Your Role (Position)\", \"profile_image\": \"https://example.com/imagens/Calasans.jpg\"}]}", false, false, false, "", true, false, true},
    {"description", "Server Description.", true, false, false, "", true, false, true},
};
constexpr size_t PUBLIC_INFORMATION_PARAMETER_COUNT = sizeof(PUBLIC_INFORMATION_PARAMETERS) / sizeof(PUBLIC_INFORMATION_PARAMETERS[0]);