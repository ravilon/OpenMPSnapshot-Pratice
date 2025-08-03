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

#include <vector>
#include <string>
#include <future>
#include <thread>
#include <algorithm>
#include <map>
#include <regex>
#include <chrono>
//
#if defined(_WIN32)
    #include <winsock2.h>
    #include <ws2tcpip.h>

    #pragma comment(lib, "ws2_32.lib")
#elif defined(__linux__)
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <cstring>
#endif
//
#include "public_ip.hpp"
#include "logger.hpp"
#include "http_client.hpp"
#include "plugin_constants.hpp"

Public_IP::Public_IP() {
    public_ip_services = {
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://icanhazip.com",
        "https://ipinfo.io/ip",
        "https://wtfismyip.com/text",
        "https://v4.ident.me",
        "https://ifconfig.me/ip",
        "https://myexternalip.com/raw",
        "https://ipecho.net/plain"
    };
}

Public_IP::~Public_IP() {}
std::string Public_IP::Resolve_Public_IP() {
    Logger::Log_Formatted(Log_Level::INFO, "Attempting to resolve server's public IP using multiple external services...");

    std::vector<std::future<std::string>> futures;
    
    for (const std::string& service_url : public_ip_services) {
        futures.push_back(std::async(std::launch::async, [this, service_url]() {
            std::string ip_result;

            try {
                std::future<std::string> fetch_future = std::async(std::launch::async, &Public_IP::Fetch_IP_From_Service, this, service_url);

                if (fetch_future.wait_for(std::chrono::seconds(IP_SERVICE_FETCH_TIMEOUT_SECONDS)) == std::future_status::timeout)
                    Logger::Log(Log_Level::WARNING, "IP service '%s' timed out after %d seconds.", service_url.c_str(), IP_SERVICE_FETCH_TIMEOUT_SECONDS);
                else
                    ip_result = fetch_future.get();
            }
            catch (const std::exception& e) {
                Logger::Log(Log_Level::WARNING, "Exception fetching IP from service '%s': %s", service_url.c_str(), e.what());
            }
            catch (...) {
                Logger::Log(Log_Level::WARNING, "Unknown exception fetching IP from service '%s'", service_url.c_str());
            }

            return ip_result;
        }));
    }

    std::map<std::string, int> ip_vote_count;
    int valid_responses_count = 0;

    for (auto& future : futures) {
        std::string ip_response = future.get();

        if (!ip_response.empty() && Is_Valid_IPv4(ip_response)) {
            ip_vote_count[ip_response]++;
            Logger::Log(Log_Level::INFO, "Resolved IP '%s' from one service. Current votes for this IP: %d.", ip_response.c_str(), ip_vote_count[ip_response]);
            valid_responses_count++;
        }
        else if (!ip_response.empty()) {
            int log_len = (std::min)(static_cast<int>(ip_response.length()), IP_SERVICE_LOG_PREVIEW_LEN);
            Logger::Log(Log_Level::WARNING, "Received invalid IP format or unexpected data from service. Raw (up to %d chars): '%.*s'. Skipping.", log_len, log_len, ip_response.substr(0, log_len).c_str());
        }
    }

    if (valid_responses_count == 0)
        return (Logger::Log_Formatted(Log_Level::ERROR_s, "No valid public IP addresses could be determined from any service."), "");

    std::string most_frequent_ip;
    int max_votes = 0;

    for (const auto& pair : ip_vote_count) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            most_frequent_ip = pair.first;
        }
    }

    double percentage_match = static_cast<double>(max_votes) / valid_responses_count * 100.0;
    Logger::Log(Log_Level::INFO, "Total valid IP responses: %d. Most frequent IP: '%s' (%d votes, %.2f%% consensus).", valid_responses_count, most_frequent_ip.c_str(), max_votes, percentage_match);

    if (percentage_match >= 50.0 || valid_responses_count == 1)
        return (Logger::Log_Formatted(Log_Level::INFO, "Public IP resolved with sufficient confidence."), most_frequent_ip);
    else {
        Logger::Log_Formatted(Log_Level::ERROR_s, "Confidence in public IP detection is too low (< 50% consensus or no single majority with multiple results). Multiple conflicting IPs detected:");

        for (const auto& pair : ip_vote_count)
            Logger::Log(Log_Level::ERROR_s, "Conflicting IP: %s (Votes: %d)", pair.first.c_str(), pair.second);
         
        return "";
    }
}

std::string Public_IP::Fetch_IP_From_Service(const std::string & url) const {
    size_t scheme_end = url.find("://");

    if (scheme_end == std::string::npos)
        return (Logger::Log(Log_Level::ERROR_s, "Invalid scheme in URL: '%s'.", url.c_str()), "");

    std::string protocol = url.substr(0, scheme_end);
    bool use_https_fetch = (protocol == "https");

    std::string host_part = url.substr(scheme_end + 3);
    size_t path_start = host_part.find("/");
    std::string host = (path_start == std::string::npos) ? host_part : host_part.substr(0, path_start);
    std::string path = (path_start == std::string::npos) ? "/" : host_part.substr(path_start);
    
    Generic_HTTP_Response response = HTTP_Client::Perform_Generic_Request(host, path, use_https_fetch, "GET", {}, {}, std::chrono::milliseconds(IP_SERVICE_REQUEST_TIMEOUT_MS));
    
    if (!response.request_ok || response.status_code < 200 || response.status_code >= 300)
        return "";

    std::string response_body_str(response.body.begin(), response.body.end());
    size_t first_char = response_body_str.find_first_not_of(" \n\r\t");

    if (std::string::npos == first_char)
        return "";
    
    size_t last_char = response_body_str.find_last_not_of(" \n\r\t");

    return response_body_str.substr(first_char, (last_char - first_char + 1));
}

std::string Public_IP::Resolve_Host_To_IP(const std::string & host_or_ip) {
    if (Is_Valid_IPv4(host_or_ip))
        return host_or_ip;

    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    std::string resolved_ip;

    int status = getaddrinfo(host_or_ip.c_str(), nullptr, &hints, &result);

    if (status != 0)
#if defined(_WIN32)
        return (Logger::Log(Log_Level::ERROR_s, "Failed to resolve host '%s' to IP. getaddrinfo error: %d", host_or_ip.c_str(), status), "");
#else
        return (Logger::Log(Log_Level::ERROR_s, "Failed to resolve host '%s' to IP. getaddrinfo error: %s (code: %d)", host_or_ip.c_str(), gai_strerror(status), status), "");
#endif
    
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> result_ptr(result, &freeaddrinfo);

    for (addrinfo* p = result_ptr.get(); p != nullptr; p = p->ai_next) {
        if (p->ai_family == AF_INET) {
            sockaddr_in* ipv4 = reinterpret_cast<sockaddr_in*>(p->ai_addr);
            char ip_buffer[INET_ADDRSTRLEN] = {0};

#if defined(_WIN32)
            if (InetNtopA(AF_INET, &(ipv4->sin_addr), ip_buffer, INET_ADDRSTRLEN)) {
                resolved_ip = ip_buffer;

                break;
            }
            else
                Logger::Log(Log_Level::WARNING, "InetNtopA failed for an IPv4 address. Error: %d", WSAGetLastError());
#else 
            if (inet_ntop(AF_INET, &(ipv4->sin_addr), ip_buffer, INET_ADDRSTRLEN)) {
                resolved_ip = ip_buffer;

                break;
            }
            else
                Logger::Log(Log_Level::WARNING, "inet_ntop failed for an IPv4 address. errno: %d (%s)", errno, strerror(errno));
#endif
        }
    }

    if (resolved_ip.empty())
        Logger::Log(Log_Level::ERROR_s, "Could not find a valid IPv4 address for host '%s' from getaddrinfo results.", host_or_ip.c_str());

    return resolved_ip;
}

bool Public_IP::Is_Valid_IPv4(std::string_view ip) {
    if (ip.empty())
        return false;

    std::regex ipv4_regex(R"(^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$)");

    return std::regex_match(ip.begin(), ip.end(), ipv4_regex);
}