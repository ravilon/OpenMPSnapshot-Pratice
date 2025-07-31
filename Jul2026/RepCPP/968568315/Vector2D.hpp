#pragma once
#include <cmath>

class Vector2D {
public:
    float x;
    float y;

    Vector2D() : x(0.0f), y(0.0f) {}
    Vector2D(float x_, float y_) : x(x_), y(y_) {}

    // Basic vector operations
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }

    Vector2D operator*(float scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }

    // Compound assignment operators
    Vector2D& operator+=(const Vector2D& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vector2D& operator-=(const Vector2D& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vector2D& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    // Length calculations for distance comparisons
    float length() const {
        return std::sqrt(x * x + y * y);
    }

    float lengthSquared() const {
        return x * x + y * y;
    }

    // Normalization for direction vectors
    Vector2D normalized() const {
        float len = length();
        if (len > 0) {
            return Vector2D(x / len, y / len);
        }
        return *this;
    }

    // Dot product for angle calculations
    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }
};
