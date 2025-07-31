///
/// \file include/utils/linearop/operator/abeltransform.hpp
/// \brief Abel transform class header
/// \details Provide the Abel transform operator
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#ifndef ASTROQUT_UTILS_OPERATOR_ABELTRANSFORM_HPP
#define ASTROQUT_UTILS_OPERATOR_ABELTRANSFORM_HPP

#include "utils/linearop/operator.hpp"

#ifdef DEBUG
#include <sstream>
#endif // DEBUG

namespace alias
{

template<class T = double>
class AbelTransform : public Operator<T>
{
private:
    size_t pic_side_;
    size_t wavelet_amount_;

public:

    /** Default constructor
     */
    AbelTransform()
        : Operator<T>()
        , pic_side_(0)
        , wavelet_amount_(0)
    {
#ifdef DEBUG
        std::cerr << "AbelTransform : Default constructor called." << std::endl;
#endif // DEBUG
    }

    /** Copy constructor
     *  \param other Object to copy from
     */
    AbelTransform(const AbelTransform& other)
        : Operator<T>(other)
        , pic_side_(other.pic_side_)
        , wavelet_amount_(other.wavelet_amount_)
    {
#ifdef DEBUG
        std::cout << "AbelTransform : Copy constructor called" << std::endl;
#endif // DEBUG
    }

    /** Move constructor
     *  \param other Object to move from
     */
    AbelTransform(AbelTransform&& other)
        : AbelTransform()
    {
#ifdef DEBUG
        std::cout << "AbelTransform : Move constructor called" << std::endl;
#endif // DEBUG
        swap(*this, other);
    }

    /** Full member constructor
     *  \param data Matrix containing the upper left component of the Abel transform.
     *  \param height Height of the full Abel matrix
     *  \param width Width of the full Abel matrix
     */
    explicit AbelTransform(Matrix<T> data, size_t height, size_t width)
        : Operator<T>(data, height, width, false)
        , pic_side_(height)
        , wavelet_amount_(height)
    {
#ifdef DEBUG
        std::cout << "AbelTransform : Full member constructor called" << std::endl;
#endif // DEBUG
    }

    /** Build constructor
     *  \brief Builds an Abel transform matrix with diagonal radius
     *  \param wavelet_amount Power of 2 amount of wavelets, typically (pixel_amount/2)^2
     *  \param pixel_amount Total amount of pixels of the target picture
     *  \param radius Amount of pixels from centre to border of galaxy, typically pixel_amount/2
     */
    explicit AbelTransform(unsigned int wavelets_amount, unsigned int pixel_amount, unsigned int radius)
        : Operator<T>(Matrix<T>((T) 0, pixel_amount/4, wavelets_amount/2), pixel_amount, wavelets_amount, false)
        , pic_side_(std::sqrt(pixel_amount))
        , wavelet_amount_(wavelets_amount)
    {
        Generate(this->data_, radius);
#ifdef DEBUG
        std::cout << "AbelTransform : Build constructor called" << std::endl;
#endif // DEBUG
    }

    /** Clone function
     *  \return A copy of the current instance
     */
    AbelTransform* Clone() const override final
    {
        return new AbelTransform(*this);
    }

    /** Default destructor
     */
    virtual ~AbelTransform()
    {
#ifdef DEBUG
        std::cout << "AbelTransform : Destructor called" << std::endl;
#endif // DEBUG
    }

    /** Valid instance test
     *  \return Throws an error message if instance is not valid.
     */
    bool IsValid() const override final
    {
        if( this->height_ != 0 && this->width_ != 0 && !this->data_.IsEmpty() )
            return true;

        throw std::invalid_argument("Operator dimensions must be non-zero and data shall not be nullptr!");
    }

    /** Swap function
     *  \param first First object to swap
     *  \param second Second object to swap
     */
    friend void swap(AbelTransform& first, AbelTransform& second) noexcept
    {
        using std::swap;

        swap(static_cast<Operator<T>&>(first), static_cast<Operator<T>&>(second));
        swap(first.pic_side_, second.pic_side_);
        swap(first.wavelet_amount_, second.wavelet_amount_);
    }

    /** Copy assignment operator
     *  \param other Object to assign to current object
     *  \return A reference to this
     */
    AbelTransform& operator=(AbelTransform other)
    {
        swap(*this, other);

        return *this;
    }


    Matrix<T> operator*(const Matrix<T>& other) const override final
    {
#ifdef DEBUG
        std::cerr << "AbelTransform: operator* called" << std::endl;
#endif // DEBUG
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

        Matrix<T> result( 0.0, this->Height(), other.Width() );

        if(!this->transposed_)
            Forward(other, result);
        else
            Transposed(other, result);

        return result;
    }

    /** Transpose in-place
     *   \return A reference to this
     */
    AbelTransform& Transpose() override final
    {
        std::swap(this->height_, this->width_);
        this->data_ = std::move(this->data_.Transpose());
        this->transposed_ = !this->transposed_;
        return *this;
    }

    /** Generate a compressed Abel matrix
     *  \brief Builds an Abel transform matrix with diagonal radius, without duplicating data or inserting zeros
     *  \param result Resulting matrix of size pixel_amount/4 * wavelets_amount/2.
     *  \param radius Amount of pixels from centre to border of galaxy, typically pixel_amount/2
     */
    void Generate(Matrix<T>& result, unsigned int radius ) const
    {
        size_t pic_side_half = pic_side_/2;
        size_t pic_side_extended = std::floor(pic_side_half*std::sqrt(2.0L));
        size_t wavelet_amount_half = wavelet_amount_/2;
        T radius_extended = radius * std::sqrt(2.0L);
        T radius_extended_to_pic_side_extended_ratio = radius_extended/(T)pic_side_extended;
        T radius_to_pic_side_ratio = radius/(T)pic_side_half;
        T radius_extended_to_wavelet_amount_half_ratio = radius_extended/(T)wavelet_amount_half;
        T* x_axis = new T[wavelet_amount_half];
        #pragma omp parallel for simd
        for( size_t i = 0; i < wavelet_amount_half; ++i )
            x_axis[i] = ((T)i+1.0L) * radius_extended_to_wavelet_amount_half_ratio;
        #pragma omp parallel for simd
        for( size_t i = 0; i < pic_side_half; ++i )
        {
            T z = (T)i * radius_to_pic_side_ratio;
            for( size_t j = 0; j < pic_side_half; ++j )
            {
                T y = (T)j * radius_extended_to_pic_side_extended_ratio;
                T s = std::sqrt(y*y + z*z);
                for(unsigned int k = 0; k < wavelet_amount_half; ++k)
                {
                    if( x_axis[k] <= s )
                        continue;

                    T ri0 = s;
                    if( k != 0 && x_axis[k-1] >= s )
                        ri0 = x_axis[k-1];

                    T ri1 = x_axis[k];
                    if( x_axis[k] < s )
                        ri1 = s;

                    size_t index = (wavelet_amount_half-i-1)*pic_side_half*wavelet_amount_half + (pic_side_half-j-1)*wavelet_amount_half + wavelet_amount_half-k-1;
                    result[index] = 2.0L*(std::sqrt(ri1*ri1 - s*s) - std::sqrt(ri0*ri0 - s*s));
                }
            }
        }
        delete[] x_axis;
    }

    /** Abel transform
     *  \brief Applies an Abel transform from the compressed Abel matrix.
     *  \param signal Signal to apply the Abel transform to. Currently only accepts double type matrix
     *  \param result Resulting matrix of size pixel_amount * wavelets_amount.
     */
    void Forward(const Matrix<T>& signal, Matrix<T>& result ) const
    {
        size_t pic_side_half = pic_side_/2;
        size_t wavelet_amount_half = wavelet_amount_/2;

#ifdef DEBUG
        int progress_step = std::max(1, (int)(pic_side_half*pic_side_half)/100);
        int step = 0;
        std::cout << std::endl;
#endif // DEBUG

        // iterating over blocks
        #pragma omp parallel for
        for( size_t block = 0; block < pic_side_half; ++block )
        {
            // iterating over rows
            for( size_t i = 0; i < pic_side_half; ++i )
            {
#ifdef DEBUG
                if( (block*pic_side_half+i) % progress_step == 0 )
                {
                    std::stringstream output;
                    output << "\r" << step++;
                    std::cout << output.str();
                }
#endif // DEBUG

                // iterating over matrix multiplication vectors
                for( size_t k = 0; k < wavelet_amount_half; ++k )
                {
                    T abel_value = this->data_[(block*pic_side_half + i)*pic_side_half + k];

                    // iterating over signal columns
                    for( size_t j = 0; j < signal.Width(); ++j )
                    {

                        // left part
                        size_t target_left_index = k*signal.Width() + j;

                        T left_temp = abel_value * signal[target_left_index];

                        size_t result_left_upper_index = (block*pic_side_ + i)*signal.Width() + j;
                        result[result_left_upper_index] += left_temp;

                        size_t result_left_lower_index = (block*pic_side_ + pic_side_ - i - 1)*signal.Width() + j;
                        result[result_left_lower_index] += left_temp;


                        // right part
                        size_t target_right_index = (pic_side_ - k - 1)*signal.Width() + j;

                        T right_temp = abel_value * signal[target_right_index];

                        size_t result_right_upper_index = ((pic_side_ - block -1)*pic_side_ + i)*signal.Width() + j;
                        result[result_right_upper_index] += right_temp;

                        size_t result_right_lower_index = ((pic_side_ - block -1)*pic_side_ + pic_side_ - i - 1)*signal.Width() + j;
                        result[result_right_lower_index] += right_temp;
                    }
                }
            }
        }
    }

    /** Transposed Abel transform
     *  \brief Applies a transposed Abel transform from the compressed Abel matrix.
     *  \param signal Signal to apply the Abel transform to. Currently only accepts double type matrix
     *  \param result Resulting matrix of size pixel_amount * wavelets_amount.
     */
    void Transposed(const Matrix<T>& signal, Matrix<T>& result ) const
    {
        size_t pic_side_half = pic_side_/2;
        size_t wavelet_amount_half = wavelet_amount_/2;

        // iterating over rows
        #pragma omp parallel for
        for( size_t i = 0; i < wavelet_amount_half; ++i )
        {

            // iterating over blocks
            for( size_t block = 0; block < pic_side_half; ++block )
            {

                // iterating over matrix multiplication vectors
                for( size_t k = 0; k < pic_side_half; ++k )
                {
                    T abel_value = this->data_[i*pic_side_half*pic_side_half + block*pic_side_half + k];

                    // iterating over signal columns
                    for( size_t j = 0; j < signal.Width(); ++j )
                    {
                        // target indices
                        size_t target_upper_left_index = (block*pic_side_ + k)*signal.Width() + j;
                        size_t target_upper_right_index = (block*pic_side_ + pic_side_ - k - 1)*signal.Width() + j;
                        size_t target_lower_left_index = ((pic_side_ - block - 1)*pic_side_ + k)*signal.Width() + j;
                        size_t target_lower_right_index = ((pic_side_ - block - 1)*pic_side_ + pic_side_ - k - 1)*signal.Width() + j;

                        // updating result
                        result[i*signal.Width() + j] += abel_value * (signal[target_upper_left_index] + signal[target_upper_right_index]);
                        result[(wavelet_amount_ - i - 1)*signal.Width() + j] += abel_value * (signal[target_lower_left_index] + signal[target_lower_right_index]);
                    }
                }
            }
        }
    }
};

} // namespace alias

#endif // ASTROQUT_UTILS_OPERATOR_ABELTRANSFORM_HPP
