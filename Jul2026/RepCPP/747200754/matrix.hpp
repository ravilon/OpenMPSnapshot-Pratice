#pragma once

#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

template<typename S>
concept KindOfSize = std::is_integral_v<S>;

template<class T>
using m_size_t = typename std::vector<T>::size_type;

template<typename T>
class Matrix {
protected:
    std::vector<T> _data = {};

    virtual T& at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept = 0;

public:
    Matrix() = default;

    Matrix(const Matrix& other) = delete;

    Matrix(Matrix&& other) noexcept = delete;

    Matrix& operator=(const Matrix& other) const = delete;

    virtual ~Matrix() = default;

    void clear() noexcept;

    [[nodiscard]]
    bool empty() const noexcept;

    template<KindOfSize... K>
    T& at(const K& ... indices) noexcept {
        return at_impl({static_cast<m_size_t<T>>(indices)...});
    }
};


template<typename T>
void Matrix<T>::clear() noexcept {
    _data.clear();
    _data.shrink_to_fit();
}

template<typename T>
bool Matrix<T>::empty() const noexcept {
    return _data.empty();
}

template<class T>
using matrix = std::shared_ptr<Matrix<T>>;

template<typename T>
class NDMatrix final : public Matrix<T> {
    using Matrix<T>::_data;
    const std::vector<m_size_t<T>> _shape = {};

    T& at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept override;

public:
    NDMatrix() = delete;

    NDMatrix(const NDMatrix& other) = delete;

    NDMatrix(NDMatrix&& other) noexcept = delete;

    NDMatrix& operator=(const NDMatrix& other) const = delete;

    ~NDMatrix() override = default;

    NDMatrix(const std::initializer_list<m_size_t<T>>& shape) noexcept: _shape(shape) {
        m_size_t<T> size = 1;
        for (const auto& dimension: _shape) {
            size *= dimension;
        }
        _data.resize(size);
    }
};

template<typename T>
T& NDMatrix<T>::at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept {
    m_size_t<T> index = 0;
    m_size_t<T> stride = 1;
    for (const auto& dimension: _shape) {
        index += stride * indices.begin()[dimension];
        stride *= _shape.cbegin()[dimension];
    }

    return _data.begin()[index];
}

template<typename T>
class TwoDMatrix final : public Matrix<T> {
    using Matrix<T>::_data;
    const m_size_t<T> _row_numbers = 0;
    const m_size_t<T> _col_numbers = 0;

    T& at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept override;

public:
    TwoDMatrix() = delete;

    TwoDMatrix(const TwoDMatrix& other) = delete;

    TwoDMatrix(TwoDMatrix&& other) noexcept = delete;

    TwoDMatrix& operator=(const TwoDMatrix& other) const = delete;

    ~TwoDMatrix() override = default;

    TwoDMatrix(
        const m_size_t<T>& row_numbers,
        const m_size_t<T>& col_numbers
    ) noexcept: _row_numbers(row_numbers), _col_numbers(col_numbers) {
        _data.resize(row_numbers * col_numbers);
    }
};

template<typename T>
T& TwoDMatrix<T>::at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept {
    const auto& i = indices.begin()[0];
    const auto& j = indices.begin()[1];
    return _data.begin()[i * _col_numbers + j];
}

template<typename T>
class ThreeDMatrix final : public Matrix<T> {
    using Matrix<T>::_data;
    const m_size_t<T> _depth = 0;
    const m_size_t<T> _row_numbers = 0;
    const m_size_t<T> _col_numbers = 0;

    T& at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept override;

public:
    ThreeDMatrix() = delete;

    ThreeDMatrix(const ThreeDMatrix& other) = delete;

    ThreeDMatrix(ThreeDMatrix&& other) noexcept = delete;

    ThreeDMatrix& operator=(const ThreeDMatrix& other) const = delete;

    ~ThreeDMatrix() override = default;

    ThreeDMatrix(
        const m_size_t<T>& depth,
        const m_size_t<T>& row_numbers,
        const m_size_t<T>& col_numbers
    ) noexcept: _depth(depth), _row_numbers(row_numbers), _col_numbers(col_numbers) {
        _data.resize(depth * row_numbers * col_numbers);
    }
};

template<typename T>
T& ThreeDMatrix<T>::at_impl(const std::initializer_list<m_size_t<T>>& indices) noexcept {
    const auto& i = indices.begin()[0];
    const auto& j = indices.begin()[1];
    const auto& k = indices.begin()[2];
    return _data.begin()[i * _row_numbers * _col_numbers + j * _col_numbers + k];
}

template<typename T, KindOfSize... S>
[[nodiscard]]
static matrix<T> make_matrix(const S& ... shape) {
    if constexpr (sizeof...(shape) == 2) {
        const auto [row_numbers, col_numbers] = std::make_tuple(shape...);
        return std::make_shared<TwoDMatrix<T>>(row_numbers, col_numbers);
    } else if constexpr (sizeof...(shape) == 3) {
        const auto [depth, row_numbers, col_numbers] = std::make_tuple(shape...);
        return std::make_shared<ThreeDMatrix<T>>(depth, row_numbers, col_numbers);
    } else {
        const std::initializer_list<m_size_t<T>> _shape = {static_cast<m_size_t<T>>(shape)...};
        return std::make_shared<NDMatrix<T>>(_shape);
    }
}
