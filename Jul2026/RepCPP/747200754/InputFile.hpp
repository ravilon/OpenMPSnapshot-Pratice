#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class InputFile final {
    std::unordered_map<std::string, std::string> pairs{};

    template<typename T>
    T get(const std::string& name, const T& dfault) const;

public:
    InputFile() = delete;

    InputFile(const InputFile& other) = delete;

    InputFile(InputFile&& other) = delete;

    InputFile& operator=(const InputFile& other) const = delete;

    explicit InputFile(const std::string& filename);

    ~InputFile() = default;

    [[nodiscard]]
    int getInt(const std::string& name, const int& dfault) const;

    [[nodiscard]]
    double getDouble(const std::string& name, const double& dfault) const;

    [[nodiscard]]
    std::string getString(const std::string& name, const std::string& dfault) const;

    [[nodiscard]]
    std::vector<double> getDoubleList(const std::string& name, const std::vector<double>& dfault) const;
};
