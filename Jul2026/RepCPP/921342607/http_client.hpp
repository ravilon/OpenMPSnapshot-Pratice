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

#include <vector>
#include <string>
#include <map>
#include <chrono>
//
#include "http_response.hpp"

struct Generic_HTTP_Response {
int status_code = 0;
std::map<std::string, std::string> headers;
std::vector<unsigned char> body;
bool request_ok = false;
};

class HTTP_Client {
public:
static HTTP_Response Send_Request(const std::string& host, const std::string& path, const std::string& data, const std::string& action, const char* api_key_for_header = nullptr);
static bool Check_Environment(const std::string& ip, const std::string& port);
static Generic_HTTP_Response Perform_Generic_Request(const std::string& host, const std::string& path, bool use_https, const std::string& method = "GET", const std::map<std::string, std::string>& headers = {}, const std::vector<unsigned char>& body = {}, const std::chrono::milliseconds& timeout = std::chrono::milliseconds(10000));
private:
#if defined(_WIN32)
static Generic_HTTP_Response Perform_Generic_Request_Platform(const std::wstring& host, const std::wstring& path, bool use_https, const std::wstring& method, const std::map<std::string, std::string>& headers, const std::vector<unsigned char>& body, const std::chrono::milliseconds& timeout);
#elif defined(__linux__)
static Generic_HTTP_Response Perform_Generic_Request_Platform(const std::string& host, const std::string& path, bool use_https, const std::string& method, const std::map<std::string, std::string>& headers, const std::vector<unsigned char>& body, const std::chrono::milliseconds& timeout);
#endif
};