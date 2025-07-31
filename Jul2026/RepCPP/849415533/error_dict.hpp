#pragma once

#include <map>
#include <string>

template <typename T> class ErrorDict {
    public:
        ErrorDict(const std::string& name);
        ErrorDict(const std::string& name,
                  std::initializer_list<std::pair<const std::string, T>> init);

        T& at(const std::string& key);
        const T& at(const std::string& key) const;

        T& operator[](const std::string& key);

    private:
        const std::string name;
        std::map<std::string, T> map;
};
