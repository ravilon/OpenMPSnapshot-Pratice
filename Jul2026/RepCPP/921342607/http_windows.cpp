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

#if defined(_WIN32)
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>
#include <sstream>
#include <windows.h>
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")
//
#include "http_client.hpp"
#include "logger.hpp"
#include "string_utils.hpp"
#include "plugin_constants.hpp"

struct WinHTTP_Handle_Closer_HTTP_Win {
void operator()(HINTERNET h) const {
if (h)
WinHttpCloseHandle(h);
}
};
using Unique_WinHTTP_Handle_HTTP_Win = std::unique_ptr<void, WinHTTP_Handle_Closer_HTTP_Win>;

Generic_HTTP_Response HTTP_Client::Perform_Generic_Request_Platform(const std::wstring& host, const std::wstring& path, bool use_https, const std::wstring& method, const std::map<std::string, std::string>& headers, const std::vector<unsigned char>& body, const std::chrono::milliseconds& timeout) {
Generic_HTTP_Response result;
result.request_ok = false;
DWORD error_code = 0;

std::wstring user_agent = String_Utils::UTF8_To_Wstring(std::string(PLUGIN_USER_AGENT_BASE) + PLUGIN_VERSION + " WinHTTP");
Unique_WinHTTP_Handle_HTTP_Win h_session(WinHttpOpen(user_agent.c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));

if (!h_session) {
error_code = GetLastError();
Logger::Log(Log_Level::WARNING, "GenericRequest: Failed WinHttpOpen. Error: %lu.", error_code);

return result;
}

DWORD timeout_ms = static_cast<DWORD>(timeout.count());

if (!WinHttpSetTimeouts(h_session.get(), timeout_ms, timeout_ms, timeout_ms, timeout_ms))
Logger::Log(Log_Level::WARNING, "GenericRequest: Failed to set WinHttp timeouts. Error: %lu.", GetLastError());

INTERNET_PORT port_to_use = use_https ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT;
Unique_WinHTTP_Handle_HTTP_Win h_connect(WinHttpConnect(h_session.get(), host.c_str(), port_to_use, 0));

if (!h_connect) {
error_code = GetLastError();
Logger::Log(Log_Level::WARNING, "GenericRequest: Failed WinHttpConnect. Error: %lu.", error_code);

return result;
}

DWORD dw_request_flags = use_https ? WINHTTP_FLAG_SECURE : 0;
Unique_WinHTTP_Handle_HTTP_Win h_request(WinHttpOpenRequest(h_connect.get(), method.c_str(), path.c_str(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, dw_request_flags));

if (!h_request) {
error_code = GetLastError();
Logger::Log(Log_Level::WARNING, "GenericRequest: Failed WinHttpOpenRequest. Error: %lu.", error_code);

return result;
}

std::wstring headers_wstr;

for (const auto& header_pair : headers)
headers_wstr += String_Utils::UTF8_To_Wstring(header_pair.first) + L": " + String_Utils::UTF8_To_Wstring(header_pair.second) + L"\r\n";

if (!headers_wstr.empty()) {
if (!WinHttpAddRequestHeaders(h_request.get(), headers_wstr.c_str(), -1L, WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE))
Logger::Log(Log_Level::WARNING, "GenericRequest: Failed to add some headers. Error: %lu", GetLastError());
}

LPCVOID request_body_ptr = body.empty() ? WINHTTP_NO_REQUEST_DATA : (LPCVOID)body.data();
DWORD request_body_size = static_cast<DWORD>(body.size());

if (!WinHttpSendRequest(h_request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0, const_cast<LPVOID>(request_body_ptr), request_body_size, request_body_size, 0)) {
error_code = GetLastError();
Logger::Log(Log_Level::WARNING, "GenericRequest: WinHttpSendRequest failed. Error: %lu.", error_code);

return result;
}

if (!WinHttpReceiveResponse(h_request.get(), NULL)) {
error_code = GetLastError();
Logger::Log(Log_Level::WARNING, "GenericRequest: WinHttpReceiveResponse failed. Error: %lu.", error_code);

return result;
}

DWORD dw_status_code = 0;
DWORD dw_size = sizeof(dw_status_code);

if (WinHttpQueryHeaders(h_request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &dw_status_code, &dw_size, WINHTTP_NO_HEADER_INDEX))
result.status_code = dw_status_code;

DWORD dw_header_size = 0;
WinHttpQueryHeaders(h_request.get(), WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, NULL, &dw_header_size, WINHTTP_NO_HEADER_INDEX);

if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
std::vector<wchar_t> header_buffer(dw_header_size / sizeof(wchar_t));

if (WinHttpQueryHeaders(h_request.get(), WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, header_buffer.data(), &dw_header_size, WINHTTP_NO_HEADER_INDEX)) {
std::wistringstream iss(std::wstring(header_buffer.data()));
std::wstring header_line;

while (std::getline(iss, header_line) && !header_line.empty()) {
size_t colon_pos = header_line.find(L':');

if (colon_pos != std::wstring::npos) {
std::wstring key_w = header_line.substr(0, colon_pos);
std::wstring value_w = header_line.substr(colon_pos + 1);

value_w.erase(0, value_w.find_first_not_of(L" \t"));
size_t last_char_pos = value_w.find_last_not_of(L"\r\n");

if (last_char_pos != std::wstring::npos)
value_w.erase(last_char_pos + 1);

result.headers[String_Utils::Wstring_To_UTF8(key_w)] = String_Utils::Wstring_To_UTF8(value_w);
}
}
}
}

if (method != L"HEAD" && result.status_code != 204) {
do {
dw_size = 0;

if (!WinHttpQueryDataAvailable(h_request.get(), &dw_size))
break;

if (dw_size == 0)
break;

std::vector<unsigned char> buffer(dw_size);
DWORD downloaded = 0;

if (!WinHttpReadData(h_request.get(), buffer.data(), dw_size, &downloaded))
break;

if (downloaded > 0)
result.body.insert(result.body.end(), buffer.begin(), buffer.begin() + downloaded);
}
while (dw_size > 0);
}

result.request_ok = true;

return result;
}
#endif