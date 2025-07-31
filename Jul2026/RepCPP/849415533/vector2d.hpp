#pragma once

#include <iostream>

class Vector2D {
    public:
        double x, y;

        Vector2D(const double x, const double y);
        Vector2D(const double n);
        Vector2D();

        Vector2D add(const Vector2D& v) const;
        Vector2D scalarProduct(const double n) const;
        double abs() const;
        double dotProduct(const Vector2D& v) const;

        Vector2D operator+(const Vector2D& v) const;
        Vector2D operator-(const Vector2D& v) const;
        double operator*(const Vector2D& v) const;
        Vector2D operator*(const double n) const;
        Vector2D operator/(const double n) const;
        double operator[](const int i) const;
};

Vector2D operator*(const double n, const Vector2D& v);
std::ostream& operator<<(std::ostream& stream, const Vector2D& v);
