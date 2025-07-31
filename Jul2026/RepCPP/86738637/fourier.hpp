///
/// \file include/utils/linearop/operator/fourier.hpp
/// \brief Fourier class header
/// \details Provide a Fourier transform operator
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#ifndef ASTROQUT_UTILS_OPERATOR_FOURIER_HPP
#define ASTROQUT_UTILS_OPERATOR_FOURIER_HPP

#include "utils/linearop/operator.hpp"

#include <complex>

namespace alias
{

inline void PrimeFactorDecomposition(int number)
{
    while (number % 2 == 0)
    {
        std::cout << 2;
        number = number/2;
    }

    for (int i = 3; i <= std::sqrt(number); i += 2)
    {
        while (number % i == 0)
        {
            std::cout << i;
            number = number/i;
        }
    }

    if (number > 2)
        std::cout << number;
}

template <class T = double>
class Fourier : public Operator<T>
{
private:
    size_t depth_max_;
    Matrix<size_t> bit_reverse_table_;
    Matrix<std::complex<T>> roots_of_unity_;
public:

    /** Default constructor
     */
    Fourier()
        : Operator<T>()
        , depth_max_(0)
        , bit_reverse_table_()
        , roots_of_unity_()
    {
#ifdef DEBUG
        std::cout << "Fourier : Default constructor called" << std::endl;
#endif // DEBUG
    }

    /** Copy constructor
     *  \param other Object to copy from
     */
    Fourier(const Fourier& other)
        : Operator<T>(other)
        , depth_max_(other.depth_max_)
        , bit_reverse_table_(other.bit_reverse_table_)
        , roots_of_unity_(other.roots_of_unity_)
    {
#ifdef DEBUG
        std::cout << "Fourier : Copy constructor called" << std::endl;
#endif // DEBUG
    }

    /** Move constructor
     *  \param other Object to move from
     */
    Fourier(Fourier&& other)
        : Fourier()
    {
#ifdef DEBUG
        std::cout << "Fourier : Move constructor called" << std::endl;
#endif // DEBUG
        swap(*this, other);
    }

    /** Build constructor
     *  \param length Length of the signal to transform, must be a power of 2
     */
    explicit Fourier(size_t length)
        : Operator<T>(length, length)
        , depth_max_(std::log2(length))
        , bit_reverse_table_(Matrix<size_t>(length, 1))
        , roots_of_unity_(Matrix<std::complex<T>>(length - 1, 1))
    {
#ifdef DEBUG
        std::cout << "Fourier : Build constructor called with length=" << length << std::endl;
#endif // DEBUG

        // build bit reverse lookup table
        for( size_t i = 0; i < length; ++i )
        {
            int num = i;
            int reverse_num = 0;
            int bit_count = depth_max_;

            while(bit_count-- > 0)
            {
                reverse_num <<= 1;
                reverse_num |= num & 1;
                num >>= 1;
            }

            bit_reverse_table_[i] = reverse_num;
        }

        // build roots of unity
        for( size_t depth = 1; depth <= depth_max_; ++depth )
        {
            double depth_length = std::pow(2, depth);
            for(size_t col = 0; col < depth_length/2; ++col)
                roots_of_unity_[std::pow(2,depth-1) - 1 + col] = std::exp( std::complex(0.0, - 2.0 * PI * col / depth_length ) );
        }
    }

    /** Clone function
     *  \return A copy of the current instance
     */
    Fourier* Clone() const override final
    {
        return new Fourier(*this);
    }

    /** Default destructor
     */
    virtual ~Fourier()
    {
#ifdef DEBUG
        std::cout << "Fourier : Destructor called" << std::endl;
#endif // DEBUG
    }

    /** Valid instance test
     *  \return Throws an error message if instance is not valid.
     */
    bool IsValid() const override final
    {
        if( this->height_ != 0 && this->width_ != 0 && depth_max_ != 0 && !bit_reverse_table_.IsEmpty() && !roots_of_unity_.IsEmpty() )
            return true;

        throw std::invalid_argument("Operator dimensions must be non-zero and function shall not be nullptr!");
    }

    /** Swap function
     *  \param first First object to swap
     *  \param second Second object to swap
     */
    friend void swap(Fourier& first, Fourier& second) noexcept
    {
        using std::swap;

        swap(static_cast<Operator<T>&>(first), static_cast<Operator<T>&>(second));
        swap(first.depth_max_, second.depth_max_);
        swap(first.bit_reverse_table_, second.bit_reverse_table_);
        swap(first.roots_of_unity_, second.roots_of_unity_);
    }

    /** Copy assignment operator
     *  \param other Object to assign to current object
     *  \return A reference to this
     */
    Fourier& operator=(Fourier other)
    {
        swap(*this, other);

        return *this;
    }

    Matrix<T> operator*(const Matrix<T>& other) const override final
    {
        std::cout << "Please use FFT and FFT2D instead." << std::endl;
        return Matrix<T>(other);
    }

    /** Fast Fourier Transform for temporary instances
     *  \brief Iterative FFT
     *  \param signal Input signal to transform with the FFT
     *  \param result Resulting matrix
     *  \author Community from https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
     *  \author Philippe Ganz <philippe.ganz@gmail.com> 2018-2019
     */
    void FFT( const Matrix<std::complex<T>>& signal, Matrix<std::complex<T>>& result ) const
    {
        // bit reversal step
        #pragma omp parallel for simd
        for( size_t i = 0; i < this->Width(); ++i )
            if( bit_reverse_table_[i] < signal.Length() )
                result[i] = signal[bit_reverse_table_[i]];

        // iterative radix-2 FFT
        for( size_t depth = 1; depth <= depth_max_; ++depth )
        {
            size_t depth_length = std::pow(2,depth);
            #pragma omp parallel for simd
            for( size_t row = 0; row < this->Width(); row += depth_length )
                for( size_t col = 0; col < depth_length/2; ++col )
                {
                    std::complex<T> e_k = result[ row + col ];
                    std::complex<T> o_k = roots_of_unity_[std::pow(2,depth-1) - 1 + col] * result[ row + col + depth_length/2 ];
                    result[ row + col ] = e_k + o_k;
                    result[ row + col + depth_length/2 ] = e_k - o_k;
                }
        }
    }

    void IFFT( const Matrix<std::complex<T>>& signal, Matrix<std::complex<T>>& result ) const
    {
        // compute a forward FFT of the signal and divide by signal length
        FFT(signal, result);
        result /= signal.Length();

        // mirror all the values except the first one
        for( size_t i = 1; i < signal.Length()/2; ++i )
            std::swap(result[i], result[signal.Length() - i]);
    }

    /** 2D Fast Fourier Transform
     *  \brief Computes 2 FFT in a row
     *  \param signal Matrix to be transformed
     *  \author Philippe Ganz <philippe.ganz@gmail.com> 2018-2019
     */
    Matrix<std::complex<T>> FFT2D( const Matrix<std::complex<T>>& signal ) const
    {
        Matrix<std::complex<T>> result(0, this->Height(), this->Width());

        // compute a 1D FFT for every row of the signal
        #pragma omp parallel for simd
        for( size_t row = 0; row < signal.Height(); ++row )
        {
            Matrix<std::complex<T>> input_row(signal.Data() + row*signal.Width(), signal.Width(), 1);
            Matrix<std::complex<T>> result_row(result.Data() + row*result.Width(), result.Width(), 1);
            FFT(input_row, result_row);
            input_row.Data(nullptr);
            result_row.Data(nullptr);
        }

        // transpose the result
        Matrix<std::complex<T>> result_transposed = result.Transpose();
        Matrix<std::complex<T>> result_final(0, this->Height(), this->Width());

        // compute a 1D FFT for every row of the transposed intermediate result, i.e. the columns of the previous FFT
        #pragma omp parallel for simd
        for( size_t row = 0; row < result_transposed.Height(); ++row )
        {
            Matrix<std::complex<T>> input_row(result_transposed.Data() + row*result_transposed.Width(), signal.Width(), 1);
            Matrix<std::complex<T>> result_row(result_final.Data() + row*result_final.Width(), result.Width(), 1);
            FFT(input_row, result_row);
            input_row.Data(nullptr);
            result_row.Data(nullptr);
        }

        // transpose back to have the original orientation
        std::move(result_final).Transpose();
        return result_final;
    }

    Matrix<std::complex<T>> IFFT2D( const Matrix<std::complex<T>>& signal ) const
    {
        Matrix<std::complex<T>> result(this->Height(), this->Width());

        // compute a 1D IFFT for every row of the signal
        #pragma omp parallel for simd
        for( size_t row = 0; row < signal.Height(); ++row )
        {
            Matrix<std::complex<T>> input_row(signal.Data() + row*signal.Width(), signal.Width(), 1);
            Matrix<std::complex<T>> result_row(result.Data() + row*result.Width(), result.Width(), 1);
            IFFT(input_row, result_row);
            input_row.Data(nullptr);
            result_row.Data(nullptr);
        }

        // transpose the result in-place
        Matrix<std::complex<T>> result_transposed = result.Transpose();
        Matrix<std::complex<T>> result_final(0, this->Height(), this->Width());

        // compute a 1D IFFT for every row of the transposed intermediate result, i.e. the columns of the previous IFFT
        #pragma omp parallel for simd
        for( size_t row = 0; row < result.Height(); ++row )
        {
            Matrix<std::complex<T>> input_row(result_transposed.Data() + row*result_transposed.Width(), signal.Width(), 1);
            Matrix<std::complex<T>> result_row(result_final.Data() + row*result_final.Width(), result.Width(), 1);
            IFFT(input_row, result_row);
            input_row.Data(nullptr);
            result_row.Data(nullptr);
        }

        // transpose back to have the original orientation
        std::move(result_final).Transpose();
        return result_final;
    }
};


} // namespace alias

#endif // ASTROQUT_UTILS_OPERATOR_FOURIER_HPP
