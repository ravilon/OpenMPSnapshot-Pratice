#pragma once

class Scheme {
public:
    Scheme() = default;

    Scheme(const Scheme& other) = delete;

    Scheme(Scheme&& other) = delete;

    Scheme& operator=(const Scheme& other) const = delete;

    virtual ~Scheme() = default;

    virtual void doAdvance(const double& dt) const = 0;

    virtual void init() const = 0;
};
