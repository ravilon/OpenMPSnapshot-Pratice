///
/// \file include/utils/linearop/matrix.hpp
/// \brief Matrix class header
/// \details Provide matrix container with multiple matrix operations used in the whole project.
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#ifndef ASTROQUT_UTILS_MATRIX_HPP
#define ASTROQUT_UTILS_MATRIX_HPP

#include "utils/linearop.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <omp.h>
#include <queue>
#include <string>
#include <type_traits>

#ifdef __unix__
#include <CCfits/CCfits>
#include <cctype>
#include <filesystem>
#endif // __unix__

namespace alias
{
template <class T> struct is_complex : std::false_type {};
template <class T> struct is_complex<std::complex<T>> : std::true_type {};

template <class T> struct ExtractType;
template <template <class ...> class Main, class Sub> struct ExtractType<Main<Sub>> { using SubType = Sub; };

/** Types of norm currently implemented
 */
enum NormType {one, two, two_squared, inf};

/** Number comparison function
 *  \param first First number to compare
 *  \param second Second number to compare
 *  \author mch from https://stackoverflow.com/a/253874/8141262
 *  \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
 */
template <class T, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
inline bool IsEqual(T first, T second)
{
    bool normal_test = std::abs(first-second) < std::abs(first+second)*std::numeric_limits<double>::epsilon()*10;
    bool subnormal_test = std::abs(first-second) < std::numeric_limits<double>::min();

    return normal_test || subnormal_test;
}
template <class T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
inline bool IsEqual(T first, T second)
{
    return first == second;
}
template <class T, typename std::enable_if_t<is_complex<T> {}>* = nullptr>
inline bool IsEqual(T first, T second)
{
    return IsEqual(std::real(first), std::real(second)) && IsEqual(std::imag(first), std::imag(second));
}
template <class T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
inline bool IsSmaller(T first, T second)
{
    return first < second;
}
template <class T, typename std::enable_if_t<is_complex<T> {}>* = nullptr>
inline bool IsSmaller(T first, T second)
{
    return std::norm(first) < std::norm(second);
}
template <class T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
inline bool IsBigger(T first, T second)
{
    return first > second;
}
template <class T, typename std::enable_if_t<is_complex<T> {}>* = nullptr>
inline bool IsBigger(T first, T second)
{
    return std::norm(first) > std::norm(second);
}

inline std::string str_tolower(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); }
                  );
    return str;
}

#ifdef VERBOSE
template <class T>
class Matrix;
template <class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat);
#endif // VERBOSE

template <class T = double>
class Matrix : public LinearOp
{
private:
    T* data_; //!< Member variable "data_"

public:
    /** Default constructor
     *  Create an empty container
     */
    Matrix() noexcept
        : data_(nullptr)
    {
#ifdef DEBUG
        std::cout << "Matrix : Default constructor called" << std::endl;
#endif // DEBUG
    }

    /** Empty constructor
     *  Create a container with default-initialized data
     *  \param height Height of the data
     *  \param width Width of the data
     */
    Matrix(size_t height, size_t width)
        : LinearOp(height, width)
        , data_(nullptr)
    {
#ifdef DEBUG
        std::cout << "Matrix : Empty constructor called with height=" << height << ", width=" << width << std::endl;
#endif // DEBUG

        if(this->length_ != 0)
            data_ = new T[this->length_];
    }

    /** Full member constructor
     *  \param data Dynamic 2D array containing the pixels
     *  \param height Height of the data
     *  \param width Width of the data
     */
    Matrix(T* data, size_t height, size_t width)
        : LinearOp(height, width)
        , data_(data)
    {
#ifdef DEBUG
        std::cout << "Matrix : Full member constructor called with data=" << data << ", height=" << height << ", width=" << width << std::endl;
#endif // DEBUG
    }

    /** Full member constructor
     *  \param data Static 2D array containing the pixels
     *  \param length Length of the static array
     *  \param height Height of the data
     *  \param width Width of the data
     */
    template <class U, typename std::enable_if_t<std::is_arithmetic<U>::value || std::is_same<T, U> {}>* = nullptr>
    Matrix(const U data[], size_t length, size_t height, size_t width)
        : Matrix(height, width)
    {
#ifdef DEBUG
        std::cout << "Matrix : Full member constructor called with data=" << data << ", length=" << length << ", height=" << height << ", width=" << width << std::endl;
#endif // DEBUG
        #pragma omp parallel for simd
        for(size_t i = 0; i < length; ++i)
            data_[i] = (T) data[i];
    }

    /** File constructor with known size
     *  \param filename Path of the binary file to read the matrix data from
     *  \param height Height of the data
     *  \param width Width of the data
     */
    template <class U = T, typename std::enable_if_t<std::is_arithmetic<U>::value>* = nullptr>
    Matrix(const std::string filename, size_t height, size_t width, U __attribute__((unused)) dummy = 0)
        : Matrix(height, width)
    {
#ifdef DEBUG
        std::cout << "Matrix : File constructor with known size called with filename=" << filename << ", height=" << height << ", width=" << width << std::endl;
#endif // DEBUG

        std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);

        size_t file_size = file.tellg();
        if(file_size == 0)
        {
            std::cerr << "Input file is empty";
            throw;
        }
        char* memblock = new char [file_size];
        file.seekg(0, std::ios::beg);
        file.read(memblock, file_size);
        file.close();

        U* reinterpret_memblock = (U*) memblock;

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] = reinterpret_memblock[i];

        delete[] memblock;
    }

    /** File constructor raw
     *  \param filename Path of the binary file to read the matrix data from
     */
    template <class U = T, typename std::enable_if_t<std::is_arithmetic<U>::value>* = nullptr>
    Matrix(const std::string filename, U __attribute__((unused)) dummy = 0)
        : data_(nullptr)
    {
#ifdef DEBUG
        std::cout << "Matrix : File constructor raw called with filename=" << filename << std::endl;
#endif // DEBUG

#ifdef __unix__
        if( str_tolower(std::filesystem::path(filename).extension()) == ".fits" )
        {
            CCfits::FITS file(filename, CCfits::Read, true);
            std::valarray<T> picture;
            file.pHDU().readAllKeys();
            file.pHDU().read(picture);

            height_ = file.pHDU().axis(0) * file.pHDU().axis(1);
            width_ = 1;
            length_ = height_;

            data_ = new T[length_];

            #pragma omp parallel for simd
            for(size_t i = 0; i < length_; ++i)
                data_[i] = picture[i];
        }
        else
#endif // __unix__
        {
            std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);

            size_t file_size = file.tellg();
            if(file_size == 0)
            {
                std::cerr << "Input file is empty";
                throw;
            }
            char* memblock = new char[file_size];
            file.seekg(0, std::ios::beg);
            file.read(memblock, file_size);
            file.close();

            U* reinterpret_memblock = (U*) memblock;

            size_t length = file_size / sizeof(U);

            data_ = new T[length];

            height_ = length;
            width_ = 1;
            length_ = height_;

            #pragma omp parallel for simd
            for(size_t i = 0; i < length_; ++i)
                data_[i] = reinterpret_memblock[i];

            delete[] memblock;
        }
    }

    /** Constant number constructor
     *  \param number Number to fill data with
     *  \param height Height of the data
     *  \param width Width of the data
     */
    template <class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
    Matrix(U number, size_t height, size_t width)
        : Matrix(height, width)
    {
#ifdef DEBUG
        std::cout << "Matrix : Constant number constructor called with number=" << number << ", height=" << height << ", width=" << width << std::endl;
#endif // DEBUG

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] = (T) number;
    }

    /** Copy constructor
     *  \param other Object to copy from
     */
    Matrix(const Matrix& other)
        : Matrix(other.height_, other.width_)
    {
#ifdef DEBUG
        std::cout << "Matrix : Copy constructor called" << std::endl;
#endif // DEBUG

        if( other.data_ != nullptr )
        {
            #pragma omp parallel for simd
            for(size_t i = 0; i < this->length_; ++i)
                data_[i] = other.data_[i];
        }
    }

    /** Move constructor
     *  \param other Object to move from
     */
    Matrix(Matrix&& other) noexcept
        : Matrix()
    {
#ifdef DEBUG
        std::cout << "Matrix : Move constructor called" << std::endl;
#endif // DEBUG

        swap(*this, other);
    }

    /** Clone function
     *  \return A copy of the current instance
     */
    Matrix Clone() const
    {
        return Matrix(*this);
    }

    /** Default destructor */
    virtual ~Matrix()
    {
#ifdef DEBUG
        std::cout << "Matrix : Destructor called" << std::endl;
#endif // DEBUG
        delete[] data_;
        data_ = nullptr;
    }

    /** Access data_
     * \return The current value of data_
     */
    T* Data() const noexcept
    {
        return data_;
    }
    /** Set data_
     * \param data New value to set
     */
    void Data(T* const data)
    {
        data_ = data;
    }

    /** Empty test operator
     *  \return True if empty.
     */
    bool IsEmpty() const noexcept
    {
        if( this->height_ == 0 && this->width_ == 0 && data_ == nullptr)
            return true;

        return false;
    }

    /** Valid instance test
     *  \return Throws an error message if instance is not valid.
     */
    bool IsValid() const override final
    {
        if( this->height_ != 0 && this->width_ != 0 && data_ != nullptr )
            return true;

        throw std::invalid_argument("Matrix dimensions must be non-zero and data shall not be empty!");
    }

    /** Negativity test
     *  \return True is a negative number is found in the array, False otherwise.
     */
    bool ContainsNeg() const noexcept
    {
#ifdef DO_ARGCHECKS
        if( this->height_ == 0 || this->width_ == 0 || data_ == nullptr)
            return false;
#endif // DO_ARGCHECKS

        for( size_t i = 0; i < this->length_; ++i )
            if( data_[i] < 0 )
                return true;

        return false;
    }

    /** Partial matrix creator
     *  \brief Generates a new matrix from the current one with data in [start,end]
     *  \param start First element of partial matrix
     *  \param end Last element of partial matrix
     *  \return A new matrix with the partial data
     */
    Matrix Partial(size_t start, size_t end) const
    {
        Matrix result(end-start+1, 1);
        std::copy(data_+start, data_+end+1, result.Data());
        return result;
    }

    /** Partial 2D matrix creator
     *  \brief Generates a new matrix from the current one with data in [(row, col), (row, col)]
     *  \param row_start Starting row, included
     *  \param row_end Last row, included
     *  \param col_start Starting column, included
     *  \param col_end Last column, included
     *  \return A new matrix with the partial data
     */
    Matrix Partial2D(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const
    {
        size_t height = row_end-row_start+1;
        size_t width = col_end-col_start+1;
        Matrix result(height, width);

        #pragma omp parallel for simd
        for(size_t row = 0; row < height; ++row)
            for(size_t col = 0; col < width; ++col)
                result[row*width + col] = data_[(row+row_start)*this->width_ + (col+col_start)];
        return result;
    }

    /** Swap function
     *  \param first First object to swap
     *  \param second Second object to swap
     */
    friend void swap(Matrix& first, Matrix& second) noexcept
    {
        using std::swap;

        swap(static_cast<LinearOp&>(first), static_cast<LinearOp&>(second));
        swap(first.data_, second.data_);
    }

    /** Copy assignment operator
     *  \param other Object to assign to current object
     *  \return A reference to this
     */
    Matrix& operator=(Matrix other)
    {
#ifdef DEBUG
        std::cout << "Matrix : Copy assignment operator called" << std::endl;
#endif // DEBUG
        swap(*this, other);
        return *this;
    }

    /** Cast operator
     *  \return A casted copy of this
     */
    template <class U, class S = T, typename std::enable_if_t<(std::is_arithmetic<S>::value && std::is_arithmetic<U>::value) ||
                                                              (std::is_arithmetic<S>::value && is_complex<U> {}) ||
                                                              (is_complex<S> {} && is_complex<U> {})                            >* = nullptr>
    operator Matrix<U>() const
    {
        Matrix<U> result(this->height_, this->width_);

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            result[i] = (U) data_[i];

        return result;
    }
    template <class U, class S = T, typename std::enable_if_t<is_complex<S> {} && std::is_arithmetic<U>::value>* = nullptr>
    operator Matrix<U>() const
    {
        Matrix<U> result(this->height_, this->width_);

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            result[i] = (U) data_[i].real();

        return result;
    }

    /** Array subscript setter operator
     *  \param index Array subscript
     *  \return A reference to the array element at index
     */
    T& operator[](size_t index) noexcept
    {
        return data_[index];
    }

    /** Array subscript getter operator
     *  \param index Array subscript
     *  \return A reference to the array element at index
     */
    template <class S = T, typename std::enable_if_t<std::is_arithmetic<S>::value>* = nullptr>
    const S operator[](size_t index) const noexcept
    {
        return data_[index];
    }
    template <class S = T, typename std::enable_if_t<is_complex<S> {}>* = nullptr>
    const S& operator[](size_t index) const noexcept
    {
        return data_[index];
    }

    /** Additive operator in-place
     *  \param other Object to add from current object
     *  \return A reference to this
     */
    Matrix& operator+=(const Matrix& other)
    {
        if( IsEmpty() )
            return *this = other;
        if( other.IsEmpty() )
            return *this;

#ifdef DO_ARGCHECKS
        try
        {
            this->ArgTest(other, add);
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            this->data_[i] += other.data_[i];

        return *this;
    }

    /** Additive operator with single number in-place
     *  \param number Number to add to current object
     *  \return A reference to this
     */
    template <class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
    Matrix& operator+=(U number)
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] += number;

        return *this;
    }

    /** Subtractive operator in-place
     *  \param other Object to subtract current object to
     *  \return A reference to this
     */
    Matrix& operator-=(const Matrix& other)
    {
        if( IsEmpty() )
            return *this = other;
        if( other.IsEmpty() )
            return *this;

#ifdef DO_ARGCHECKS
        try
        {
            this->ArgTest(other, add);
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            this->data_[i] -= other.data_[i];

        return *this;
    }

    /** Subtractive operator with single number in-place
     *  \param number Number to remove from current object
     *  \return A reference to this
     */
    template <class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
    Matrix& operator-=(U number)
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] -= number;

        return *this;
    }

    /** Multiplicative operator in-place
     *  \param other Object to multiply to current object
     *  \return A reference to this
     */
    Matrix& operator*=(const Matrix& other)
    {
        *this = *this * other;
        return *this;
    }

    /** Multiplicative operator
     *  \param other Object to multiply current object with
     *  \return A new instance containing the result
     */
    Matrix operator*(const Matrix& other) const
    {
        if( IsEmpty() )
            return Matrix(other);
        if( other.IsEmpty() )
            return *this;

#ifdef DO_ARGCHECKS
        try
        {
            this->ArgTest(other, mult);
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        Matrix result(this->height_, other.width_);

        // other is a vector
        if( other.height_ == 1 || other.width_ == 1 )
        {
            // this is a vector
            if( this->height_ == 1 || this->width_ == 1 )
            {
                std::cerr << "Vector times a vector is a single number, consider using the Inner function instead" << std::endl;
                result.data_[0] = Inner(*this, other);
            }
            // this is a matrix
            else
            {
                MatrixVectorMult(*this, other, result);
            }
        }
        // other is a matrix
        else
        {
            // this is a vector
            if( this->height_ == 1 || this->width_ == 1 )
            {
                VectorMatrixMult(*this, other, result);
            }
            // this is a matrix
            else
            {
                MatrixMatrixMult(*this, other, result);
            }
        }

        return result;
    }

    /** Multiplicative operator with single number in-place
     *  \param number Number to multiply current object with
     *  \return A reference to this
     */
    template <class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
    Matrix& operator*=(U number)
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] *= number;

        return *this;
    }

    /** Element-wise multiply in-place
     *  \param other Object to multiply to current object, element-wise
     *  \return A reference to this
     */
    Matrix& operator&=(const Matrix& other)
    {
        if( IsEmpty() )
            return *this = other;
        if( other.IsEmpty() )
            return *this;

#ifdef DO_ARGCHECKS
        try
        {
            this->ArgTest(other, element_wise);
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] *= other.data_[i];

        return *this;
    }

    /** Divide operator with single number in-place
     *  \param number Number to divide the current object with
     *  \return A reference to this
     */
    template <class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
    Matrix& operator/=(U number)
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] /= number;

        return *this;
    }

    /** Element-wise divide operator in-place
     *  \param other Object to divide from current object, element-wise
     *  \return A reference to this
     */
    Matrix& operator/=(const Matrix& other)
    {
        if( IsEmpty() )
            return *this = other;
        if( other.IsEmpty() )
            return *this;
#ifdef DO_ARGCHECKS
        try
        {
            this->ArgTest(other, element_wise);
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] /= other.data_[i];

        return *this;
    }

    /** Transpose
    *   \return A new instance with the transposed matrix
    */
    Matrix Transpose() const &
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        Matrix result(this->width_, this->height_); // width <--> height

        // vector
        if( this->height_ == 1 || this->width_ == 1 )
        {
            if( omp_get_max_threads() > 1)
            {
                #pragma omp parallel for simd
                for(size_t i = 0; i < this->length_; ++i)
                    result[i] = data_[i];
            }
            else
            {
                std::copy(data_, data_ + (this->length_), result.data_);
            }
        }
        // matrix
        else
        {
            #pragma omp parallel for
            for ( size_t i = 0; i < this->height_; ++i )
            {
                #pragma omp simd
                for ( size_t j = 0 ; j < this->width_; ++j )
                    result.data_[ j * this->height_ + i ] = data_[ i * this->width_ + j ];
            }
        }
        return result;
    }

    /** Transpose in-place
    *   \return A reference to this
    */
    Matrix&& Transpose() &&
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        // vector
        if( this->height_ == 1 || this->width_ == 1 )
        {
            std::swap(this->height_, this->width_);
        }
        // matrix
        else
        {
            // square matrix, simple swap
            if( this->height_ == this->width_ )
            {

                #pragma omp parallel for
                for ( size_t i = 0; i < this->height_; ++i )
                {
                    #pragma omp simd
                    for ( size_t j = i+1 ; j < this->width_; ++j )
                        std::swap(data_[i * this->width_ + j], data_[j * this->width_ + i]);
                }
                std::swap(this->height_, this->width_); // width <--> height
            }

            // rect matrix, cyclic permutations
            // TODO in-place algorithm
            // for now, simple copy to new array
            else
            {
                Matrix result(this->width_, this->height_); // width <--> height

                #pragma omp parallel for
                for ( size_t i = 0; i < this->height_; ++i )
                {
                    #pragma omp simd
                    for ( size_t j = 0 ; j < this->width_; ++j )
                        result.data_[ j * this->height_ + i ] = data_[ i * this->width_ + j ];
                }
                *this = std::move(result);
            }
        }
        return std::move(*this);
    }

    /** Log operator in-place
     *  Applies the log function to all positive elements, replace by zero otherwise
     *  \return Current object overwritten with the result
     */
    Matrix&& Log() &&
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
        {
            // real part of log of negative numbers is 0
            data_[i] = ((data_[i] >= 0) ? std::log(data_[i]) : 0);
        }

        return std::move(*this);
    }

    /** Log operator
     *  Applies the log function to all positive elements, replace by zero otherwise
     *  \return A new instance containing the result
     */
    Matrix Log() const &
    {
        return Matrix(*this).Log();
    }

    /** Abs operator in-place
     *  Applies the abs function to all elements
     *  \return Current object overwritten with the result
     */
    Matrix&& Abs() &&
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for(size_t i = 0; i < this->length_; ++i)
            data_[i] = std::abs(data_[i]);

        return std::move(*this);
    }

    /** Abs operator
     *  Applies the abs function to all elements
     *  \return A new instance containing the result
     */
    Matrix Abs() const &
    {
        return Matrix(*this).Abs();
    }

    /** Shrinkage in-place
     *   Apply the shrinkage algorithm
     *   \param thresh_factor The threshold factor to be used on the data
     *   \return A reference to this
     */
    Matrix&& Shrink(double thresh_factor) &&
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for( size_t i = 0; i < this->length_; ++i )
            data_[i] *= std::max(1 - thresh_factor / (double)std::abs(data_[i]), 0.0);

        return std::move(*this);
    }

    /** Shrinkage
     *   Apply the shrinkage algorithm
     *   \param thresh_factor The threshold factor to be used on the data
     *   \return A new instance containing the result
     */
    Matrix Shrink(double thresh_factor) const &
    {
        return Matrix(*this).Shrink(thresh_factor);
    }

    /** Remove negative values in-place
     *   Set all values below zero in [first, last) to zero
     *   \param first First element of the range
     *   \param last Last element of the range
     *   \return A reference to this
     */
    Matrix&& RemoveNeg(size_t first, size_t last) &&
    {
        #pragma omp parallel for simd
        for(size_t i = first; i < last; ++i)
            if(data_[i] < 0)
                data_[i] = (T)0;

        return std::move(*this);
    }

    /** Remove negative values at location
     *   Set all values below zero in [first, last) to zero
     *   \param first First element of the range
     *   \param last Last element of the range
     *   \return A new instance containing the result
     */
    Matrix RemoveNeg(size_t first, size_t last) const &
    {
        return Matrix(*this).RemoveNeg(first, last);
    }

    /** Remove negative values at location in-place
     *   Set values below zero to zero
     *   \param indices An array of index of location to remove negative values
     *   \return A reference to this
     */
    Matrix&& RemoveNeg(Matrix<size_t> indices) &&
    {
        #pragma omp parallel for simd
        for(size_t i = 0; i < indices.Length(); ++i)
            if(data_[indices[i]] < 0)
                data_[indices[i]] = (T)0;

        return std::move(*this);
    }

    /** Remove negative values at location
     *   Set values below zero to zero
     *   \param indices An array of index of location to remove negative values
     *   \return A new instance containing the result
     */
    Matrix RemoveNeg(Matrix<size_t> indices) const &
    {
        return Matrix(*this).RemoveNeg(indices);
    }

    /** Connected components
     *   Returns a matrix with connected components reduced to the single maximum element or every connected component
     *   \return A new instance containing the result
     */
    Matrix ConnectedComponentsMax() const &
    {
#ifdef VERBOSE
        std::cout << *this;
#endif // VERBOSE
        size_t CC_total = 0;
        Matrix<T> CC_max(0.0, height_, width_);
        Matrix<bool> CC_marked(false, height_, width_);

        for(size_t row = 0; row < height_; ++row)
        {
            for(size_t col = 0; col < width_; ++col)
            {
                // consider only non-zero data and point not already marked
                if( ! IsEqual(data_[row*width_ + col], (T) 0.0) && ! CC_marked[row*width_ + col])
                {
                    // this is a new connected component
                    ++CC_total;

                    // mark current point
                    CC_marked[row*width_ + col] = true;

                    // initialise the new CC with the current non-zero point
                    std::vector<size_t> CC_current = {row*width_ + col};
                    std::queue<std::pair<int, int>> points;
                    points.push(std::pair<int, int>(row, col));
#ifdef VERBOSE
                    std::cout << "(" << row << "," << col << ") <=> " << row*width_ + col << std::endl;
#endif // VERBOSE

                    // Breadth First Search to look for all points connected to the currentCC.
                    while( ! points.empty() )
                    {
                        std::pair<int, int> point = points.front();
                        points.pop();
                        // check all 8 neighbours of the current point
                        for(int neighbor_row = std::max(point.first-1, 0); neighbor_row < std::min(point.first+2, (int)width_); ++neighbor_row )
                        {
                            for(int neighbor_col = std::max(point.second-1, 0); neighbor_col < std::min(point.second+2, (int)height_); ++neighbor_col )
                            {
                                size_t point_index = neighbor_row*width_ + neighbor_col;
                                // do not check if it is already marked
                                if(! CC_marked[point_index])
                                {
                                    CC_marked[point_index] = true;
                                    // consider only if non zero
                                    if( ! IsEqual( data_[point_index], (T) 0.0) )
                                    {
                                        CC_current.push_back(point_index);
                                        points.push(std::pair<int, int>(neighbor_row, neighbor_col));
#ifdef VERBOSE
                                        std::cout << "(" << neighbor_row << "," << neighbor_col << ") <=> " << neighbor_row*width_ + neighbor_col << std::endl;
#endif // VERBOSE
                                    }
                                }
                            }
                        }
                    }

                    // get maximum value of the connected component
                    size_t max_index = CC_current[0];
                    for(size_t i = 1; i < CC_current.size(); ++i)
                        if(IsSmaller(data_[max_index], data_[CC_current[i]]))
                            max_index = CC_current[i];
#ifdef VERBOSE
                    std::cout << "Max index: " << max_index << std::endl;
#endif // VERBOSE

                    // assign the maximum value to its index in the result
                    CC_max[max_index] = data_[max_index];

#ifdef VERBOSE
                    std::cout << CC_max;
#endif // VERBOSE
                }
            }
        }

        std::cout << CC_total << " connected components found in point source matrix. Keeping only maximum value." << std::endl;

        return CC_max;
    }

    /** Get indices of non zero elements
     *   \return A new instance containing the result
     */
    Matrix<size_t> NonZeroIndices() const &
    {
        std::vector<size_t> indices;
        for(size_t i = 0; i < this->length_; ++i)
            if(data_[i] < (T)0 || data_[i] > (T)0)
                indices.push_back(i);

        return Matrix<size_t>(&indices[0], indices.size(), indices.size(), 1);
    }

    /** Get indices of zero elements
     *   \return A new instance containing the result
     */
    Matrix<size_t> ZeroIndices() const &
    {
        std::vector<size_t> indices;
        for(size_t i = 0; i < this->length_; ++i)
            if( IsEqual(data_[i], (T)0) )
                indices.push_back(i);

        return Matrix<size_t>(&indices[0], indices.size(), indices.size(), 1);
    }

    /** Get amount of non zero elements
     *   \return A new instance containing the result
     */
    size_t NonZeroAmount() const &
    {
        size_t result = 0;
        #pragma omp parallel for reduction(+:result)
        for(size_t i = 0; i < this->length_; ++i)
            if(data_[i] < 0 || data_[i] > 0)
                ++result;

        return result;
    }

    /** Padding function for temporary instances
     *  \brief Pads the matrix with zero, if necessary, up to size height * width
     *  \param S Type of the resulting Matrix, defaults to same type
     *  \param height Height of the resulting padded Matrix
     *  \param width Width of the resulting padded Matrix
     *  \return A reference to this if same size, a reference to the result else
     */
    template <class U>
    Matrix<U>&& Padding(size_t height, size_t width) &&
    {
        // check if padding is necessary
        bool height_padding = false;
        bool width_padding = false;
        if( height_ < height )
            height_padding = true;
        if( width_ < width )
            width_padding = true;

        // no padding necessary
        if( !height_padding && !width_padding )
        {
            // no type conversion necessary
            if( std::is_same<T,U>::value )
                return std::forward(*this);
            Matrix<U> result = *this;
            return std::move(result);
        }

        Matrix<U> result(height, width);

        // copy the data into the resulting Matrix and set the rest to zero
        if( width_padding )
        {
            #pragma omp parallel for
            for( size_t i = 0; i < height_; ++i )
            {
                std::copy( data_ + i*width_, data_ + (i+1)*width_, result.Data() + i*width );
                std::fill( result.Data() + i*width + width_, result.Data() + (i+1)*width, (T) 0 );
            }
        }

        // set the last rows to zero
        if( height_padding )
        {
            #pragma omp parallel for
            for( size_t i = height_; i < height; ++i )
                std::fill( result.Data() + i*width, result.Data() + (i+1)*width, (T) 0 );
        }

        return std::move(result);
    }

    /** Norm
     *  Norm of all elements considered as a one dimensional vector
     * \return The result of type T
     */
    double Norm(const NormType l_norm) const
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        switch(l_norm)
        {
        case one:
        {
            double result = 0.0;
            #pragma omp parallel for reduction(+:result)
            for(size_t i = 0; i < this->length_; ++i)
                result += std::abs(data_[i]);
            return result;
        }
        case two:
        {
            return std::sqrt(Norm(two_squared));
        }
        case two_squared:
        {
            double result = 0.0;
            #pragma omp parallel for reduction(+:result)
            for(size_t i = 0; i < this->length_; ++i)
                result += std::norm(data_[i]);
            return result;
        }
        case inf:
        {
            double result = 0.0;
            #pragma omp parallel for reduction(max:result)
            for(size_t i = 0; i < this->length_; ++i)
                result = std::max(result, std::abs(data_[i]));
            return result;
        }
        default:
        {
            return 0.0L;
        }
        }
    }

    /** Sum
     *  Sum of all elements, considered as a one dimensional vector
     *  \return The result of type T
     */
    template <class U = T, typename std::enable_if_t<std::is_arithmetic<U>::value>* = nullptr>
    T Sum() const
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        T result = 0;
        #pragma omp parallel for reduction(+:result)
        for(size_t i = 0; i < this->length_; ++i)
            result += data_[i];
        return result;
    }
    template <class U = T, typename std::enable_if_t<is_complex<U> {}>* = nullptr, class ComplexSubType = typename ExtractType<U>::SubType>
    T Sum() const
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        double result_real = 0;
        double result_imag = 0;
        #pragma omp parallel for reduction(+:result_real, result_imag)
        for(size_t i = 0; i < this->length_; ++i)
        {
            result_real += data_[i].real();
            result_imag += data_[i].imag();
        }
        return std::complex<ComplexSubType>(result_real, result_imag);
    }

    /** Min
     *  Min of all elements, considered as a one dimensional vector
     *  \return The result of type T
     */
    T Min() const
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        T result = std::numeric_limits<T>::infinity();
        #pragma omp parallel for reduction(min:result)
        for(size_t i = 0; i < this->length_; ++i)
            result = std::min(result, data_[i]);
        return result;

    }

    /** Max
     *  Max of all elements, considered as a one dimensional vector
     *  \return The result of type T
     */
    T Max() const
    {
#ifdef DO_ARGCHECKS
        try
        {
            IsValid();
        }
        catch (const std::exception&)
        {
            throw;
        }
#endif // DO_ARGCHECKS

        T result = -std::numeric_limits<T>::infinity();
        #pragma omp parallel for reduction(max:result)
        for(size_t i = 0; i < this->length_; ++i)
            result = std::max(result, data_[i]);
        return result;
    }

    void PrintRefQual() const &
    {
        std::cout << "I'm an lvalue !" << std::endl;
    }
    void PrintRefQual() &&
    {
        std::cout << "I'm an rvalue !" << std::endl;
    }
};

/** Comparison operator equal
 *  \param first First matrix of comparison
 *  \param second Second matrix of comparison
 *  \return True if both object are the same pointer-wise
 */
template <class T>
bool operator==(const Matrix<T>& first, const Matrix<T>& second)
{
    if( first.Height() != second.Height() || first.Width() != second.Width() || first.Data() != second.Data() )
        return false;

    return true;
}

/** Comparison operator not-equal
 *  \param first First matrix of comparison
 *  \param second Second matrix of comparison
 *  \return False if both object are the same element-wise, True else
 */
template <class T>
bool operator!=(const Matrix<T>& first, const Matrix<T>& second)
{
    return !(first == second);
}

/** Element-wise comparison operator
 *  \param first First matrix of comparison
 *  \param second Second matrix of comparison
 *  \return True if both object are the same element-wise, False else
 */
template <class T>
bool Compare(const Matrix<T>& first, const Matrix<T>& second)
{
    bool result = true;
    size_t amount_wrong = 0;

    for(size_t i = 0; i < first.Length(); ++i)
    {
        if( !IsEqual(first[i], second[i]) )
        {
            {
                result = false;
                ++amount_wrong;
#ifdef VERBOSE
                std::cout << std::setprecision(16) << i << " : " << first[i] << " != " << second[i] << std::endl;
#endif // VERBOSE
            }
        }
    }

    std::cout << "Amount wrong : " << amount_wrong << " / " << first.Length();
    std::cout << std::defaultfloat << " ~= " << 100.0 * (double)amount_wrong/(double)first.Length() << "%" << std::endl;

    return result;
}

/** Additive operator, both Matrix are lvalues
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to add to current object, lvalue ref
 *  \return A new instance containing the result
 */
template<class T>
Matrix<T> operator+(const Matrix<T>& first, const Matrix<T>& second)
{
    Matrix<T> result(second);
    return result += first;
}

/** Additive operator, first Matrix is an rvalue
 *  \param first Matrix, rvalue ref
 *  \param second Matrix to add to current object, lvalue ref
 *  \return A reference to second
 */
template<class T>
Matrix<T> operator+(Matrix<T>&& first, const Matrix<T>& second)
{
    return std::move(first += second); // if first is temporary, no need to allocate new memory for the result
}

/** Additive operator, second Matrix is an rvalue
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to add to current object, rvalue ref
 *  \return A reference to second
 */
template<class T>
Matrix<T> operator+(const Matrix<T>& first, Matrix<T>&& second)
{
    return std::move(second += first); // if second is temporary, no need to allocate new memory for the result
}

/** Additive operator with single number
 *  \param number Number to add to current object
 *  \param mat Matrix, lvalue ref
 *  \return A reference to this
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator+(const Matrix<T>& mat, U number)
{
    Matrix<T> result(mat);
    return result += number;
}

/** Additive operator with single number of temporary instance
 *  \param number Number to add to current object
 *  \param mat Matrix, rvalue ref
 *  \return A reference to this
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator+(Matrix<T>&& mat, U number)
{
    return std::move(mat += number);
}

/** Subtractive operator, both Matrix are lvalues
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to subtract from current object, lvalue ref
 *  \return A new instance containing the result
 */
template<class T>
Matrix<T> operator-(const Matrix<T>& first, const Matrix<T>& second)
{
    Matrix<T> result(first);
    return result -= second;
}

/** Subtractive operator, first Matrix is an rvalue, second an lvalue
 *  \param first Matrix, rvalue ref
 *  \param second Matrix to subtract from current object, lvalue ref
 *  \return A reference to first
 */
template<class T>
Matrix<T> operator-(Matrix<T>&& first, const Matrix<T>& second)
{
    return std::move(first -= second); // if first is temporary, no need to allocate new memory for the result
}

/** Subtractive operator, first Matrix is an lvalue, second an rvalue
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to subtract from current object, rvalue ref
 *  \return A reference to second
 */
template<class T>
Matrix<T> operator-(const Matrix<T>& first, Matrix<T>&& second)
{
    // Need to implement it fully again since `-` is not commutative

#ifdef DO_ARGCHECKS
    try
    {
        first.ArgTest(second, add);
    }
    catch (const std::exception&)
    {
        throw;
    }
#endif // DO_ARGCHECKS

    #pragma omp parallel for simd
    for(size_t i = 0; i < first.Length(); ++i)
        second[i] = first[i] - second[i];

    return std::move(second); // if second is temporary, no need to allocate new memory for the result
}

/** Subtractive operator with single number
 *  \param number Number to remove from current object
 *  \param mat Matrix, lvalue ref
 *  \return A reference to this
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator-(const Matrix<T>& mat, U number)
{
    Matrix<T> result(mat);
    return result -= number;
}

/** Subtractive operator with single number of temporary instance
 *  \param number Number to remove from current object
 *  \param mat Matrix, rvalue ref
 *  \return A reference to this
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator-(Matrix<T>&& mat, U number)
{
    return std::move(mat -= number);
}

/** Multiplicative operator element-wise
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to multiply the current object with
 *  \return A new instance containing the result
 */
template <class T>
Matrix<T> operator&(const Matrix<T>& first, const Matrix<T>& second)
{
    Matrix<T> result(first);
    return result &= second;
}

/** Multiplicative operator element-wise of temporary instance
 *  \param first Matrix, rvalue ref
 *  \param second Matrix to multiply the current object with, lvalue ref
 *  \return A reference to first
 */
template <class T>
Matrix<T> operator&(Matrix<T>&& first, const Matrix<T>& second)
{
    return std::move(first &= second);
}

/** Multiplicative operator element-wise of temporary instance
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to multiply the current object with, rvalue ref
 *  \return A reference to second
 */
template <class T>
Matrix<T> operator&(const Matrix<T>& first, Matrix<T>&& second)
{
    return std::move(second &= first);
}

/** Multiplicative operator with single number right
 *  \param mat Matrix, lvalue ref
 *  \param number Number to multiply the current object with
 *  \return A new instance containing the result
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator*(const Matrix<T>& mat, U number)
{
    Matrix<T> result(mat);
    return result *= number;
}

/** Multiplicative operator with single number right of temporary instance
 *  \param mat Matrix, rvalue ref
 *  \param number Number to multiply the current object with
 *  \return A reference to this
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator*(Matrix<T>&& mat, U number)
{
    return std::move(mat *= number);
}

/** Multiplicative operator with single number left
 *  \param number Number to multiply the current object with
 *  \param mat Matrix, lvalue ref
 *  \return A new instance containing the result
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator*(U number, const Matrix<T>& mat)
{
    return mat * number;
}

/** Multiplicative operator with single number left of temporary instance
 *  \param number Number to multiply the current object with
 *  \param mat Matrix, rvalue ref
 *  \return A reference to this
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator*(U number, Matrix<T>&& mat)
{
    return mat * number;
}

/** Divide operator with single number
 *  \param mat Matrix, lvalue ref
 *  \param number Number to divide the current object with
 *  \return A new instance containing the result
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator/(const Matrix<T>& mat, U number)
{
    Matrix<T> result(mat);
    return result /= number;
}

/** Divide operator with single number of temporary instance
 *  \param mat Matrix, rvalue ref
 *  \param number Number to divide the current object with
 *  \return A reference to mat
 */
template <class T, class U, typename std::enable_if_t<std::is_arithmetic<U>::value || is_complex<U> {}>* = nullptr>
Matrix<T> operator/(Matrix<T>&& first, U number)
{
    return std::move(first /= number);
}

/** Divide operator element-wise
 *  \param first Matrix, lvalue ref
 *  \param second Matrix to divide the current object with
 *  \return A new instance containing the result
 */
template <class T>
Matrix<T> operator/(const Matrix<T>& first, const Matrix<T>& second)
{
    Matrix<T> result(first);
    return result /= second;
}

/** Divide operator element-wise of temporary instance
 *  \param first Matrix, rvalue ref
 *  \param second Matrix to divide the current object with
 *  \return A reference to first
 */
template <class T>
Matrix<T> operator/(Matrix<T>&& first, const Matrix<T>& second)
{
    return std::move(first /= second);
}

/** Output stream operator
 *  \param os The output stream to write to
 *  \param mat Matrix to read from
 *  \return A reference to os
 */
template <class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat)
{
#ifdef DO_ARGCHECKS
    try
    {
        mat.IsValid();
    }
    catch (const std::exception&)
    {
        throw;
    }
#endif // DO_ARGCHECKS

    size_t width = sizeof(T) * 1.5;

    for( size_t i = 0; i < mat.Height(); ++i )
    {
        os << std::endl;
        for( size_t j = 0; j < mat.Width(); ++j )
            os << std::setw(width) << mat[i*mat.Width() + j] << " ";
    }
    os << std::endl;
    return os;
}

/** Output file operator
 *  \param filename The name of the file to write to
 *  \param mat Matrix to read from
 */
template <class T>
void operator<<(std::string filename, const Matrix<T>& mat)
{
#ifdef DO_ARGCHECKS
    try
    {
        mat.IsValid();
    }
    catch (const std::exception&)
    {
        throw;
    }
#endif // DO_ARGCHECKS

#ifdef __unix__
    if( str_tolower(std::filesystem::path(filename).extension()) == ".fits" )
    {
        filename = "!" + filename;
        long axes[2] = { (long)mat.Width(), (long)mat.Height() };
        CCfits::FITS file(filename, DOUBLE_IMG, 2, axes);
        std::valarray<T> mat_array(mat.Length());
        #pragma omp parallel for simd
        for(size_t i = 0; i < mat.Length(); ++i)
            mat_array[i] = mat[i];

        file.pHDU().write(1, mat.Length(), mat_array);
    }
    else
#endif // __unix__
    {
        std::ofstream file(filename, std::ios::binary | std::ios::out);

        T* memblock = new T[mat.Length()];
        #pragma omp parallel for simd
        for(size_t i = 0; i < mat.Length(); ++i)
            memblock[i] = mat[i];
        char* reinterpret_memblock = (char*) memblock;
        file.write(reinterpret_memblock, mat.Length()*sizeof(T));

        file.close();
        delete[] memblock;
    }
}

/** Inner product
 *  \brief The inner product of two vectors, i.e. first^T * second
 *  \param first Vector
 *  \param second Vector
 *  \return The result of type T
 */
template <class T>
T Inner(const Matrix<T>& first, const Matrix<T>& second)
{
#ifdef DO_ARGCHECKS
    try
    {
        first.ArgTest(second, add);
    }
    catch (const std::exception&)
    {
        throw;
    }
#endif // DO_ARGCHECKS

    T result = 0;
    #pragma omp parallel for reduction(+:result)
    for(size_t i = 0; i < first.Length(); ++i)
        result += first[i] * second[i];
    return result;
}
inline std::complex<double> Inner(const Matrix<std::complex<double>>& first, const Matrix<std::complex<double>>& second)
{
#ifdef DO_ARGCHECKS
    try
    {
        first.ArgTest(second, add);
    }
    catch (const std::exception&)
    {
        throw;
    }
#endif // DO_ARGCHECKS

    double result_real = 0;
    double result_imag = 0;
    #pragma omp parallel for reduction(+:result_real, result_imag)
    for(size_t i = 0; i < first.Length(); ++i)
    {
        std::complex<double> local_result = std::conj(first[i]) * second[i];
        result_real += local_result.real();
        result_imag += local_result.imag();
    }
    return std::complex<double>(result_real, result_imag);
}

/** Matrix Matrix multiplication
 *  Performs a matrix-matrix multiplication : result = first * second
 *  \param first First matrix of size l by m
 *  \param second Second matrix of size m by n
 *  \param result Resulting matrix of size l by n
 */
template <class T>
static inline void MatrixMatrixMult(const Matrix<T>& first, const Matrix<T>& second, Matrix<T>& result)
{
    // Init result to zero
    #pragma omp parallel for simd
    for(size_t i = 0; i < result.Length(); ++i)
        result[i] = 0;
    #pragma omp parallel for
    for(size_t i = 0; i < first.Height(); ++i)
        for(size_t k = 0; k < first.Width(); ++k)
            for(size_t j = 0; j < second.Width(); ++j)
                result[i*second.Width() + j] += first[i*first.Width() + k] * second[k*second.Width() + j];
}

/** Matrix Vector multiplication
 *  Performs a matrix-vector multiplication : result = mat * vect
 *  \param mat Matrix of size m by n
 *  \param vect Vector of size n by 1
 *  \param result Resulting vector of size m by 1
 *  \param type Computation strategy to use
 */
template <class T>
static inline void MatrixVectorMult(const Matrix<T>& mat, const Matrix<T>& vect, Matrix<T>& result)
{
    #pragma omp parallel for
    for(size_t i = 0; i < mat.Height(); ++i)
    {
        result[i] = (T)0;
        for(size_t j = 0; j < mat.Width(); ++j)
            result[i] += mat[i*mat.Width() + j] * vect[j];
    }
}

/** Vector Matrix multiplication
 *  Performs a matrix-vector multiplication : result = vect * mat
 *  \param vect Vector of size 1 by m
 *  \param mat Matrix of size m by n
 *  \param result Resulting vector of size 1 by n
 */
template <class T>
static inline void VectorMatrixMult(const Matrix<T>& vect, const Matrix<T>& mat, Matrix<T>& result)
{
    // Init result to zero
    #pragma omp parallel for simd
    for(size_t i = 0; i < result.Length(); ++i)
        result[i] = (T)0;
    #pragma omp parallel for
    for(size_t i = 0; i < mat.Height(); ++i)
        for(size_t j = 0; j < mat.Width(); ++j)
            result[j] += vect[i] * mat[i*mat.Width() + j];
}

} // namespace alias

#endif // ASTROQUT_UTILS_MATRIX_HPP
