///
/// \file include/utils/linearop/operator/blurring.hpp
/// \brief Blurring operator class header
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#ifndef ASTROQUT_UTILS_OPERATOR_BLUR_HPP
#define ASTROQUT_UTILS_OPERATOR_BLUR_HPP

#ifdef BLURRING_CONVOLUTION
#include "utils/linearop/operator/convolution.hpp"
#else
#include "utils/linearop/operator/fourier.hpp"
#endif // BLURRING_CONVOLUTION


namespace alias
{

template<class T = double>
class Blurring : public Operator<T>
{
public:
#ifdef BLURRING_CONVOLUTION
    Convolution<T> convolution_;
#else
    size_t filter_size_;
    Fourier<T> fourier_;
    Matrix<std::complex<T>> filter_freq_domain_;
#endif // BLURRING_CONVOLUTION

public:

    /** Default constructor
     */
    Blurring()
        : Operator<T>()
#ifdef BLURRING_CONVOLUTION
        , convolution_()
#else
        , filter_size_(0)
        , fourier_()
        , filter_freq_domain_()
#endif // BLURRING_CONVOLUTION
    {
#ifdef DEBUG
        std::cout << "Blurring : Default constructor called" << std::endl;
#endif // DEBUG
    }

    /** Copy constructor
     *  \param other Object to copy from
     */
    Blurring(const Blurring& other)
        : Operator<T>(other)
#ifdef BLURRING_CONVOLUTION
        , convolution_(other.convolution_)
#else
        , filter_size_(other.filter_size_)
        , fourier_(other.fourier_)
        , filter_freq_domain_(other.filter_freq_domain_)
#endif // BLURRING_CONVOLUTION
    {
#ifdef DEBUG
        std::cout << "Blurring : Copy constructor called" << std::endl;
#endif // DEBUG
    }

    /** Move constructor
     *  \param other Object to move from
     */
    Blurring(Blurring&& other)
        : Blurring()
    {
#ifdef DEBUG
        std::cout << "Blurring : Move constructor called" << std::endl;
#endif // DEBUG
        swap(*this, other);
    }

    /** Generate constructor
     *  \param threshold Threshold for blurring mask size
     *  \param R0 Core radius of the PSF
     *  \param alpha Decrease speed of the PSF
     *  \param pic_size Size of the picture the operator is acting on
     */
    explicit Blurring(T threshold, T R0, T alpha, size_t pic_size)
        : Operator<T>(pic_size, pic_size)
#ifdef BLURRING_CONVOLUTION
        , convolution_(Generate(threshold, R0, alpha))
#else
        , filter_size_(0)
        , fourier_()
        , filter_freq_domain_()
#endif // BLURRING_CONVOLUTION
    {
#ifdef DEBUG
        std::cout << "Blurring : Generate constructor called with threshold=" << threshold << ", R0=" << R0 << ", alpha=" << alpha << ", pic_size=" << pic_size << std::endl;
#endif // DEBUG

#ifndef BLURRING_CONVOLUTION
        Matrix<T> filter = Generate(threshold, R0, alpha);

        *this = Blurring(filter, pic_size);
#endif // BLURRING_CONVOLUTION
    }

    /** Full member constructor
     *  \param data Blurring filter matrix
     *  \param pic_size Size of the picture the operator is acting on
     */
    explicit Blurring(const Matrix<T>& filter, size_t pic_size)
        : Operator<T>(pic_size, pic_size)
#ifdef BLURRING_CONVOLUTION
        , convolution_(filter)
#else
        , filter_size_(filter.Width())
        , fourier_(std::pow(2,std::ceil(std::log2(pic_size + filter_size_ - 1))))
        , filter_freq_domain_()
#endif // BLURRING_CONVOLUTION
    {
#ifdef DEBUG
        std::cout << "Blurring : Full member constructor called with filter=" << &filter << ", pic_size=" << pic_size << std::endl;
#endif // DEBUG

#ifndef BLURRING_CONVOLUTION
        filter_freq_domain_ = fourier_.FFT2D(filter);
#endif // BLURRING_CONVOLUTION
    }

    /** File constructor
     *  \brief Loads the blurring filter from a file
     *  \param path Path to the blurring filter
     *  \param pic_size Width of the target picture
     */
    explicit Blurring(const std::string& path, size_t pic_size)
        : Operator<T>(pic_size, pic_size)
#ifdef BLURRING_CONVOLUTION
        , convolution_()
#else
        , filter_size_(0)
        , fourier_()
        , filter_freq_domain_()
#endif // BLURRING_CONVOLUTION
    {
#ifdef DEBUG
        std::cout << "Blurring : File constructor called with path=" << path << ", pic_size=" << pic_size << std::endl;
#endif // DEBUG

        // load raw data from file
        Matrix<T> filter(path);

        // determine the filter's size
        size_t filter_size = std::sqrt(filter.Length());
        filter.Height(filter_size);
        filter.Width(filter_size);

        *this = Blurring(filter, pic_size);
    }

    /** Clone function
     *  \return A copy of the current instance
     */
    Blurring* Clone() const override final
    {
        return new Blurring(*this);
    }

    /** Default destructor
     */
    virtual ~Blurring()
    {
#ifdef DEBUG
        std::cout << "Blurring : Destructor called" << std::endl;
#endif // DEBUG
    }

    /** Valid instance test
     *  \return Throws an error message if instance is not valid.
     */
    bool IsValid() const override final
    {
        if( this->height_ != 0 && this->width_ != 0 &&
#ifdef BLURRING_CONVOLUTION
            convolution_.IsValid() )
#else
            filter_size_ != 0 &&
            fourier_.IsValid() &&
            filter_freq_domain_.IsValid() )
#endif // BLURRING_CONVOLUTION
            return true;

        throw std::invalid_argument("Blurring dimensions must be non-zero and members shall be valid!");
    }

    /** Generate a blurring filter
     *  \brief Builds a blurring filter to be used in a convolution
     *  \param threshold Threshold for blurring mask size
     *  \param R0 Core radius of the PSF
     *  \param alpha Decrease speed of the PSF
     */
    Matrix<T> Generate(T threshold, T R0, T alpha )
    {
        T radius_squared = R0*R0;
        int mask_size = std::ceil(std::sqrt((std::pow(threshold, -1.0/alpha)-1.0)*radius_squared));
        Matrix<double> result(2*mask_size+1, 2*mask_size+1);

        #pragma omp parallel for
        for(int i = -mask_size; i <= mask_size; ++i)
        {
            T i_squared = i*i;
            #pragma omp simd
            for(int j = -mask_size; j <= mask_size; ++j)
            {
                T j_squared = j*j;
                result[(i+mask_size)*(2*mask_size+1) + (j+mask_size)] = std::pow(1 + (i_squared + j_squared)/radius_squared, -alpha);
            }
        }
        result /= result.Sum();

        return result;
    }

    /** Swap function
     *  \param first First object to swap
     *  \param second Second object to swap
     */
    friend void swap(Blurring& first, Blurring& second) noexcept
    {
        using std::swap;

        swap(static_cast<Operator<T>&>(first), static_cast<Operator<T>&>(second));

#ifdef BLURRING_CONVOLUTION
        swap(first.convolution_, second.convolution_);
#else
        swap(first.filter_size_, second.filter_size_);
        swap(first.fourier_, second.fourier_);
        swap(first.filter_freq_domain_, second.filter_freq_domain_);
#endif // BLURRING_CONVOLUTION
    }

    /** Copy assignment operator
     *  \param other Object to assign to current object
     *  \return A reference to this
     */
    Blurring& operator=(Blurring other)
    {
        swap(*this, other);

        return *this;
    }

    Matrix<T> operator*(const Matrix<T>& other) const
    {
#ifdef DEBUG
        std::cerr << "Blurring: operator* called" << std::endl;
#endif // DEBUG
#ifdef DO_ARGCHECKS
        if( !IsValid() || !other.IsValid() || other.Height() == 1 || other.Width() == 1 )
        {
            throw std::invalid_argument("Can not apply the blurring with these Matrices.");
        }
#endif // DO_ARGCHECKS

#ifdef BLURRING_CONVOLUTION
        return convolution_ * other;
#else
        Matrix<std::complex<T>> other_freq_domain = fourier_.FFT2D(other);
        Matrix<std::complex<T>> full_result = fourier_.IFFT2D( filter_freq_domain_ & other_freq_domain );
        Matrix<T> result(other.Height(), other.Width());
        size_t filter_offset = (filter_size_ - 1) / 2;

        #pragma omp parallel for simd collapse(2)
        for(size_t row = 0; row < other.Height(); ++row)
            for(size_t col = 0; col < other.Width(); ++col)
                result[row*other.Width() + col] = full_result[(row+filter_offset)*full_result.Width() + (col+filter_offset)].real();

        return result;
#endif // BLURRING_CONVOLUTION

    }
};

} // namespace alias

#endif // ASTROQUT_UTILS_OPERATOR_BLUR_HPP
