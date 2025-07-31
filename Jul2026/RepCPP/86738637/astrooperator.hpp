///
/// \file include/utils/linearop/operator/astrooperator.hpp
/// \brief Combination of all operators to create the main operator
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#ifndef ASTROQUT_UTILS_OPERATOR_ASTROOPERATOR_HPP
#define ASTROQUT_UTILS_OPERATOR_ASTROOPERATOR_HPP

#include "utils/linearop/operator/abeltransform.hpp"
#include "utils/linearop/operator/blurring.hpp"
#include "utils/linearop/operator/matmult/spline.hpp"
#include "utils/linearop/operator/wavelet.hpp"
#include "WS/astroQUT.hpp"

namespace alias
{

template<class T = double>
class AstroOperator : public Operator<T>
{
private:
    size_t pic_size_;
    AbelTransform<T> abel_;
    Blurring<T> blurring_;
    Matrix<T> sensitivity_;
    Matrix<T> standardize_;
    Spline<T> spline_;
    Wavelet<T> wavelet_;

public:
    /** Default constructor
     */
    AstroOperator()
        : Operator<T>()
        , pic_size_()
        , abel_()
        , blurring_()
        , sensitivity_()
        , standardize_()
        , spline_()
        , wavelet_()
    {
#ifdef DEBUG
        std::cout << "AstroOperator : Default constructor called" << std::endl;
#endif // DEBUG
    }

    /** Copy constructor
     *  \param other Object to copy from
     */
    AstroOperator(const AstroOperator& other)
        : Operator<T>(other)
        , pic_size_(other.pic_size_)
        , abel_(other.abel_)
        , blurring_(other.blurring_)
        , sensitivity_(other.sensitivity_)
        , standardize_(other.standardize_)
        , spline_(other.spline_)
        , wavelet_(other.wavelet_)
    {
#ifdef DEBUG
        std::cout << "AstroOperator : Copy constructor called" << std::endl;
#endif // DEBUG
    }

    /** Move constructor
     *  \param other Object to move from
     */
    AstroOperator(AstroOperator&& other)
        : AstroOperator()
    {
#ifdef DEBUG
        std::cout << "AstroOperator : Move constructor called" << std::endl;
#endif // DEBUG
        swap(*this, other);
    }

    /** Build constructor
     *  \brief Builds the AstroOperator
     *  \param pic_size Side size of the picture in pixel
     *  \param wavelet_amount
     *  \param radius
     *  \param sensitivity
     *  \param standardize
     *  \param params
     *  \param transposed
     */
    explicit AstroOperator(size_t pic_size,
                           size_t wavelet_amount,
                           size_t radius,
                           const Matrix<T> sensitivity,
                           const Matrix<T> standardize,
                           bool transposed = false,
                           WS::Parameters<T> params = WS::Parameters<T>() )
        : Operator<T>(Matrix<T>(),
                      transposed ? (pic_size+2)*pic_size : pic_size*pic_size,
                      transposed ? pic_size*pic_size : (pic_size+2)*pic_size,
                      transposed)
        , pic_size_(pic_size)
        , abel_(transposed ?
                AbelTransform<T>(wavelet_amount, pic_size*pic_size, radius).Transpose() :
                AbelTransform<T>(wavelet_amount, pic_size*pic_size, radius))
        , blurring_(Blurring<T>(params.blurring_filter, pic_size))
        , sensitivity_(sensitivity)
        , standardize_(standardize)
        , spline_(transposed ?
                  Spline<T>(pic_size).Transpose() :
                  Spline<T>(pic_size))
        , wavelet_(transposed ?
                   Wavelet<T>((WaveletType) params.wavelet[0], params.wavelet[1]).Transpose() :
                   Wavelet<T>((WaveletType) params.wavelet[0], params.wavelet[1]))
    {
#ifdef DEBUG
        std::cout << "AstroOperator : Build constructor called" << std::endl;
#endif // DEBUG
    }

    /** Full member constructor
     *  \brief Constructs the AstroOperator with all member attributes given
     *  \param pic_size Side size of the picture in pixel
     *  \param wavelet_amount
     *  \param radius
     *  \param sensitivity
     *  \param standardize
     *  \param params
     *  \param transposed
     */
    explicit AstroOperator(size_t pic_size,
                           const AbelTransform<T> abel,
                           const Blurring<T> blurring,
                           const Matrix<T> sensitivity,
                           const Matrix<T> standardize,
                           const Spline<T> spline,
                           const Wavelet<T> wavelet,
                           bool transposed = false )
        : Operator<T>(Matrix<T>(),
                      transposed ? (pic_size+2)*pic_size : pic_size*pic_size,
                      transposed ? pic_size*pic_size : (pic_size+2)*pic_size,
                      transposed)
        , pic_size_(pic_size)
        , abel_( transposed ? abel : abel.Transpose() )
        , blurring_(blurring)
        , sensitivity_(sensitivity)
        , standardize_(standardize)
        , spline_( transposed ? spline : spline.Transpose() )
        , wavelet_( transposed ? wavelet : wavelet.Transpose() )
    {
#ifdef DEBUG
        std::cout << "AstroOperator : Full member constructor called" << std::endl;
#endif // DEBUG
    }

    /** Default destructor
     */
    virtual ~AstroOperator()
    {
#ifdef DEBUG
        std::cout << "AstroOperator : Destructor called" << std::endl;
#endif // DEBUG
    }

    /** Clone function
     *  \return A copy of the current instance
     */
    AstroOperator* Clone() const override final
    {
        return new AstroOperator(*this);
    }

    size_t PicSize() const
    {
        return pic_size_;
    }
    void PicSize(size_t pic_size)
    {
        pic_size_ = pic_size;
    }

    AbelTransform<T> Abel() const
    {
        return abel_;
    }
    void Abel(const AbelTransform<T> abel)
    {
        abel_ = abel;
    }

    Blurring<T> Blur() const
    {
        return blurring_;
    }
    void Blur(const Blurring<T> blurring)
    {
        blurring_ = blurring;
    }

    Matrix<T> Sensitivity() const
    {
        return sensitivity_;
    }
    void Sensitivity(const Matrix<T> sensitivity)
    {
        sensitivity_ = sensitivity;
    }

    Matrix<T> Standardize() const
    {
        return standardize_;
    }
    void Standardize(const Matrix<T> standardize)
    {
        standardize_ = standardize;
    }

    Spline<T> SplineOp() const
    {
        return spline_;
    }
    void SplineOp(const Spline<T> spline)
    {
        spline_ = spline;
    }

    Wavelet<T> WaveletOp() const
    {
        return wavelet_;
    }
    void WaveletOp(const Wavelet<T> wavelet)
    {
        wavelet_ = wavelet;
    }

    /** Transpose in-place
     *  \return A reference to this
     */
    AstroOperator& Transpose() override final
    {
        this->transposed_ = !this->transposed_;
        std::swap(this->height_, this->width_);
        abel_ = abel_.Transpose();
        spline_ = spline_.Transpose();
        wavelet_ = wavelet_.Transpose();
        return *this;
    }

    /** Swap function
     *  \param first First object to swap
     *  \param second Second object to swap
     */
    friend void swap(AstroOperator& first, AstroOperator& second) noexcept
    {
        using std::swap;

        swap(static_cast<Operator<T>&>(first), static_cast<Operator<T>&>(second));
        swap(first.pic_size_, second.pic_size_);
        swap(first.abel_, second.abel_);
        swap(first.blurring_, second.blurring_);
        swap(first.sensitivity_, second.sensitivity_);
        swap(first.standardize_, second.standardize_);
        swap(first.spline_, second.spline_);
        swap(first.wavelet_, second.wavelet_);
    }

    /** Copy assignment operator
     *  \param other Object to assign to current object
     *  \return A reference to this
     */
    AstroOperator& operator=(AstroOperator other)
    {
        swap(*this, other);

        return *this;
    }

    virtual Matrix<T> operator*(const Matrix<T>& other) const override final
    {
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
        Matrix<T> result;
        if(!this->transposed_)
        {
            result = BAW(other);
        }
        else
        {
            result = WtAtBt(other);
        }
        return result;
    }

    Matrix<T> BAW(const Matrix<T> source,
                  bool standardize = true,
                  bool apply_wavelet = true,
                  bool apply_spline = true,
                  bool ps = true ) const
    {
#ifdef DEBUG
        std::cerr << "BAW called" << std::endl;
#endif // DEBUG
#ifdef DO_ARGCHECKS
        if( this->transposed_ )
        {
            std::cerr << "You shall not use BAW on transposed operator!" << std::endl;
            throw;
        }
#endif // DO_ARGCHECKS

        Matrix<T> normalized_source = source;
        if(standardize)
            normalized_source /= standardize_;

        // W * xw
        Matrix<T> result_wavelet;
        if( apply_wavelet )
        {
            Matrix<T> source_wavelet(&normalized_source[0], pic_size_, 1);
            result_wavelet = wavelet_ * source_wavelet;
            source_wavelet.Data(nullptr);
        }

        // W * xs
        Matrix<T> result_spline;
        if( apply_spline )
        {
            Matrix<T> source_spline(&normalized_source[pic_size_], pic_size_, 1);
            result_spline = spline_ * source_spline;
            source_spline.Data(nullptr);
        }

        // A * (Wxw + Wxs)
        Matrix<T> result = abel_ * (result_wavelet + result_spline);
        result.Height(pic_size_);
        result.Width(pic_size_);

        // AWx + ps
        if( ps )
        {
            Matrix<T> source_ps(&normalized_source[2*pic_size_], pic_size_, pic_size_);
            result += source_ps;
            source_ps.Data(nullptr);
        }

        // B(AWx + ps)
        result = blurring_ * result;
        result.Height(pic_size_*pic_size_);
        result.Width(1);

        // E' .* B(AWx + ps)
        result = result & sensitivity_;

#ifdef DEBUG
        std::cerr << "BAW done" << std::endl;
#endif // DEBUG
        return result;
    }

    Matrix<T> WtAtBt(const Matrix<T> source,
                     bool standardize = true,
                     bool apply_wavelet = true,
                     bool apply_spline = true,
                     bool ps = true ) const
    {
#ifdef DEBUG
        std::cerr << "WtAtBt called" << std::endl;
#endif // DEBUG
#ifdef DO_ARGCHECKS
        if( ! this->transposed_ )
        {
            std::cerr << "You shall not use WtAtBt on a not transposed operator!" << std::endl;
            throw;
        }
#endif // DO_ARGCHECKS
        // result matrix
        Matrix<T> result((apply_wavelet + apply_spline + ps*pic_size_)*pic_size_, 1);

        // E' .* x
        Matrix<T> BEtx = source & sensitivity_;
        BEtx.Height(pic_size_);
        BEtx.Width(pic_size_);

        // B * Etx
        BEtx = blurring_ * BEtx;
        BEtx.Height(pic_size_*pic_size_);
        BEtx.Width(1);

        if( ps )
        {
            #pragma omp parallel for simd
            for(size_t i = 0; i < pic_size_*pic_size_; ++i)
                result[(apply_wavelet + apply_spline)*pic_size_ + i] = BEtx[i];
        }

        // A' * BEtx
        Matrix<T> AtBEtx = abel_ * BEtx;

        // W' * AtBEtx
        if( apply_wavelet )
        {
            Matrix<T> result_wavelet = wavelet_ * AtBEtx;
            #pragma omp parallel for simd
            for(size_t i = 0; i < pic_size_; ++i)
                result[i] = result_wavelet[i];
        }

        // S' * AtBEtx
        if( apply_spline )
        {
            Matrix<T> result_spline = spline_ * AtBEtx;
            #pragma omp parallel for simd
            for(size_t i = 0; i < pic_size_; ++i)
                result[apply_wavelet*pic_size_ + i] = result_spline[i];
        }

        // standardize
        if(standardize)
            result /= standardize_;
#ifdef DEBUG
        std::cerr << "WtAtBt done" << std::endl;
#endif // DEBUG
        return result;
    }
};

} // namespace alias

#endif // ASTROQUT_UTILS_OPERATOR_ASTROOPERATOR_HPP
