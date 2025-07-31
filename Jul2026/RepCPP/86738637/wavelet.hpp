///
/// \file include/utils/linearop/operator/wavelet.hpp
/// \brief Wavelet transform class header
/// \details Provide the Wavelet transform operator
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#ifndef ASTROQUT_UTILS_OPERATOR_WAVELET_HPP
#define ASTROQUT_UTILS_OPERATOR_WAVELET_HPP

#include "utils/linearop/operator.hpp"

namespace alias
{

enum WaveletType {haar, beylkin, coiflet, daubechies, symmlet, vaidyanathan, battle};
enum FilterType {low, high};

template<class T = double>
class Wavelet : public Operator<T>
{
private:
    WaveletType wavelet_type_;
    int parameter_;
    Matrix<T> low_pass_filter_;
    Matrix<T> high_pass_filter_;

public:

    /** Default constructor
     */
    Wavelet()
        : Operator<T>()
        , wavelet_type_((WaveletType) 0)
        , parameter_(0)
        , low_pass_filter_(Matrix<T>())
        , high_pass_filter_(Matrix<T>())
    {
#ifdef DEBUG
        std::cout << "Wavelet : Default constructor called" << std::endl;
#endif // DEBUG
    }

    /** Copy constructor
     *  \param other Object to copy from
     */
    Wavelet(const Wavelet& other)
        : Operator<T>(other)
        , wavelet_type_(other.wavelet_type_)
        , parameter_(other.parameter_)
        , low_pass_filter_(other.low_pass_filter_)
        , high_pass_filter_(other.high_pass_filter_)
    {
#ifdef DEBUG
        std::cout << "Wavelet : Copy constructor called" << std::endl;
#endif // DEBUG
    }

    /** Move constructor
     *  \param other Object to move from
     */
    Wavelet(Wavelet&& other)
        : Wavelet()
    {
#ifdef DEBUG
        std::cout << "Wavelet : Move constructor called" << std::endl;
#endif // DEBUG
        swap(*this, other);
    }

    /** Full member constructor
     *  \param low_pass_filter QMF matrix for low pass filtering
     *  \param low_pass_filter Mirrored QMF matrix for high pass filtering
     *  \param wavelet_type Wavelet type, can be one of haar, beylkin, coiflet, daubechies, symmlet, vaidyanathan, battle
     *  \param parameter Integer parameter specific to each wavelet type
     *  \param transposed Transposition state of the operator
     */
    explicit Wavelet(Matrix<T>&& low_pass_filter, Matrix<T>&& high_pass_filter, WaveletType wavelet_type, int parameter, bool transposed = false)
        : Operator<T>(Matrix<T>(), 1, 1, transposed)
        , wavelet_type_(wavelet_type)
        , parameter_(parameter)
        , low_pass_filter_(low_pass_filter)
        , high_pass_filter_(high_pass_filter)
    {
#ifdef DEBUG
        std::cout << "Wavelet : Full member constructor called" << std::endl;
#endif // DEBUG
    }

    /** Build constructor
     *  \brief Builds the Wavelet operator with the qmf matrix corresponding to type and parameter
     *  \param wavelet_type Wavelet type, can be one of haar, beylkin, coiflet, daubechies, symmlet, vaidyanathan, battle
     *  \param parameter Integer parameter specific to each wavelet type
     *  \param transposed Transposition state of the operator
     */
    explicit Wavelet(WaveletType wavelet_type, int parameter, bool transposed = false)
        : Operator<T>(Matrix<T>(), 1, 1, transposed)
        , wavelet_type_(wavelet_type)
        , parameter_(parameter)
        , low_pass_filter_(MakeONFilter(wavelet_type, parameter, low))
        , high_pass_filter_(MakeONFilter(wavelet_type, parameter, high))
    {
#ifdef DEBUG
        std::cout << "Wavelet : Build constructor called" << std::endl;
#endif // DEBUG
    }

    /** Clone function
     *  \return A copy of the current instance
     */
    Wavelet* Clone() const override final
    {
        return new Wavelet(*this);
    }

    /** Default destructor
     */
    virtual ~Wavelet()
    {
#ifdef DEBUG
        std::cout << "Wavelet : Destructor called" << std::endl;
#endif // DEBUG
    }

    /** Valid instance test
     *  \return Throws an error message if instance is not valid.
     */
    bool IsValid() const override final
    {
        if( !low_pass_filter_.IsEmpty() && !high_pass_filter_.IsEmpty() )
        {
            return true;
        }
        else
        {
            throw std::invalid_argument("Filters shall not be empty!");
        }
    }

    /** Swap function
     *  \param first First object to swap
     *  \param second Second object to swap
     */
    friend void swap(Wavelet& first, Wavelet& second) noexcept
    {
        using std::swap;

        swap(static_cast<Operator<T>&>(first), static_cast<Operator<T>&>(second));
        swap(first.wavelet_type_, second.wavelet_type_);
        swap(first.parameter_, second.parameter_);
        swap(first.low_pass_filter_, second.low_pass_filter_);
        swap(first.high_pass_filter_, second.high_pass_filter_);
    }

    /** Copy assignment operator
     *  \param other Object to assign to current object
     *  \return A reference to this
     */
    Wavelet& operator=(Wavelet other)
    {
        swap(*this, other);

        return *this;
    }

    Matrix<T> operator*(const Matrix<T>& other) const override final
    {
#ifdef DEBUG
        std::cerr << "Wavelet: operator* called" << std::endl;
#endif // DEBUG
#ifdef DO_ARGCHECKS
        if( !this->IsValid() || !other.IsValid() )
            throw;
#endif // DO_ARGCHECKS

        Matrix<T> result( other.Height(), other.Width() );
        T* temp_1 = new T[other.Height()];
        T* temp_2 = new T[other.Height()];

        if(!this->transposed_)
        {
            IWT_PO(other, result, 0, 0, temp_1, temp_2);
        }
        else
        {
            FWT_PO(other, result, 0, 0, temp_1, temp_2);
        }

        delete[] temp_1;
        delete[] temp_2;

        return result;
    }

    /** Transpose in-place
     *   \return A reference to this
     */
    virtual Wavelet& Transpose() override final
    {
        std::swap(this->height_, this->width_);
        this->transposed_ = !this->transposed_;
        return *this;
    }

    /** Make orthonormal  QMF filter
     *  \brief Generate orthonormal QMF filter for Wavelet Transform
     *  \param wavelet_type Wavelet type, can be one of haar, beylkin, coiflet, daubechies, symmlet, vaidyanathan, battle
     *  \param parameter Integer parameter specific to each wavelet type
     *  \param filter_type Low or high pass filter
     *  \author Jonathan Buckheit and David Donoho, MATLAB version in Wavelab 850, 1993-1995
     *  \author Philippe Ganz <philippe.ganz@gmail.com> 2018
     */
    Matrix<T> MakeONFilter(WaveletType wavelet_type,
                           int parameter,
                           FilterType filter_type) const
    {
        size_t data_size = 0;
        T data[59] = {(T)0};

        switch(wavelet_type)
        {
        case haar:
        {
            data_size = 2;
            T data_local[2] = {(T)1/std::sqrt((T)2), (T)1/std::sqrt((T)2)};
            std::copy(data_local, data_local+data_size, data);
            break;
        }

        case beylkin:
        {
            data_size = 18;
            T data_local[18] = {0.099305765374, 0.424215360813, 0.699825214057, 0.449718251149, -0.110927598348, -0.264497231446, 0.026900308804, 0.155538731877, -0.017520746267, -0.088543630623, 0.019679866044, 0.042916387274, -0.017460408696, -0.014365807969, 0.010040411845, 0.001484234782, -0.002736031626, 0.000640485329};
            std::copy(data_local, data_local+data_size, data);
            break;
        }
        case coiflet:
        {
            switch(parameter)
            {
            case 1:
            {
                data_size = 6;
                T data_local[6] = {0.038580777748, -0.126969125396, -0.077161555496, 0.607491641386, 0.745687558934, 0.226584265197};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 2:
            {
                data_size = 12;
                T data_local[12] = {0.016387336463, -0.041464936782, -0.067372554722, 0.386110066823, 0.81272363545, 0.417005184424, -0.076488599078, -0.059434418646, 0.023680171947, 0.005611434819, -0.001823208871, -0.000720549445};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 3:
            {
                data_size = 18;
                T data_local[18] = {-0.003793512864, 0.007782596426, 0.023452696142, -0.065771911281, -0.061123390003, 0.40517690241, 0.793777222626, 0.428483476378, -0.071799821619, -0.082301927106, 0.034555027573, 0.015880544864, -0.009007976137, -0.002574517688, 0.001117518771, 0.00046621696, -0.000070983303, -0.000034599773};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 4:
            {
                data_size = 24;
                T data_local[24] = {0.000892313668, -0.001629492013, -0.007346166328, 0.016068943964, 0.026682300156, -0.08126669968, -0.056077313316, 0.41530840703, 0.78223893092,0.434386056491, -0.066627474263, -0.096220442034, 0.039334427123, 0.025082261845, -0.015211731527, -0.005658286686, 0.003751436157, 0.001266561929, -0.000589020757, -0.000259974552, 0.000062339034, 0.000031229876, -0.00000325968, -0.000001784985};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 5:
            {
                data_size = 30;
                T data_local[30] = {-0.000212080863, 0.000358589677, 0.002178236305, -0.004159358782, -0.010131117538, 0.023408156762, 0.028168029062, -0.091920010549, -0.052043163216, 0.421566206729, 0.77428960374, 0.437991626228, -0.062035963906, -0.105574208706, 0.041289208741, 0.032683574283, -0.019761779012, -0.009164231153, 0.006764185419, 0.002433373209, -0.001662863769, -0.000638131296, 0.00030225952, 0.000140541149, -0.000041340484, -0.000021315014, 0.000003734597, 0.000002063806, -0.000000167408, -0.000000095158};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            default:
            {
                break;
            }
            }
            break;
        }
        case daubechies:
        {
            switch(parameter)
            {
            case 4:
            {
                data_size = 4;
                T data_local[4] = {0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 6:
            {
                data_size = 6;
                T data_local[6] = {0.33267055295, 0.806891509311, 0.459877502118, -0.13501102001, -0.085441273882, 0.035226291882};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 8:
            {
                data_size = 8;
                T data_local[8] = {0.230377813309, 0.714846570553, 0.63088076793, -0.027983769417, -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 10:
            {
                data_size = 10;
                T data_local[10] = {0.160102397974, 0.603829269797, 0.724308528438, 0.138428145901, -0.242294887066, -0.032244869585, 0.07757149384, -0.006241490213, -0.012580751999, 0.003335725285};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 12:
            {
                data_size = 12;
                T data_local[12] = {0.11154074335, 0.494623890398, 0.751133908021, 0.315250351709, -0.226264693965, -0.129766867567, 0.097501605587, 0.02752286553, -0.031582039317, 0.000553842201, 0.004777257511, -0.001077301085};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 14:
            {
                data_size = 14;
                T data_local[14] = {0.077852054085, 0.396539319482, 0.729132090846, 0.469782287405, -0.143906003929, -0.224036184994, 0.071309219267, 0.080612609151, -0.038029936935, -0.016574541631, 0.012550998556, 0.000429577973, -0.001801640704, 0.000353713800};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 16:
            {
                data_size = 16;
                T data_local[16] = {0.054415842243, 0.312871590914, 0.675630736297, 0.585354683654, -0.015829105256, -0.284015542962, 0.000472484574, 0.12874742662, -0.017369301002, -0.044088253931, 0.013981027917, 0.008746094047, -0.004870352993, -0.000391740373, 0.000675449406, -0.000117476784};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 18:
            {
                data_size = 18;
                T data_local[18] = {0.038077947364, 0.243834674613, 0.60482312369, 0.657288078051, 0.133197385825, -0.293273783279, -0.096840783223, 0.148540749338, 0.030725681479, -0.067632829061, 0.000250947115, 0.022361662124, -0.004723204758, -0.004281503682, 0.001847646883, 0.000230385764, -0.000251963189, 0.000039347320};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 20:
            {
                data_size = 20;
                T data_local[20] = {0.026670057901, 0.188176800078, 0.527201188932, 0.688459039454, 0.281172343661, -0.249846424327, -0.195946274377, 0.127369340336, 0.093057364604, -0.071394147166, -0.029457536822, 0.033212674059, 0.003606553567, -0.010733175483, 0.001395351747, 0.001992405295, -0.000685856695, -0.000116466855, 0.00009358867, -0.000013264203};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            default:
            {
                break;
            }
            }
            break;
        }
        case symmlet:
        {
            switch(parameter)
            {
            case 4:
            {
                data_size = 8;
                T data_local[8] = {-0.107148901418, -0.041910965125, 0.703739068656, 1.136658243408, 0.421234534204, -0.140317624179, -0.017824701442, 0.045570345896};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 5:
            {
                data_size = 10;
                T data_local[10] = {0.038654795955, 0.041746864422, -0.055344186117, 0.281990696854, 1.023052966894, 0.89658164838, 0.023478923136, -0.247951362613, -0.029842499869, 0.027632152958};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 6:
            {
                data_size = 12;
                T data_local[12] = {0.021784700327, 0.004936612372, -0.166863215412, -0.068323121587, 0.694457972958, 1.113892783926, 0.477904371333, -0.102724969862, -0.029783751299, 0.06325056266, 0.002499922093, -0.011031867509};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 7:
            {
                data_size = 14;
                T data_local[14] = {0.003792658534, -0.001481225915, -0.017870431651, 0.043155452582, 0.096014767936, -0.070078291222, 0.024665659489, 0.758162601964, 1.085782709814, 0.408183939725, -0.198056706807, -0.152463871896, 0.005671342686, 0.014521394762};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 8:
            {
                data_size = 16;
                T data_local[16] = {0.002672793393, -0.0004283943, -0.021145686528, 0.005386388754, 0.069490465911, -0.038493521263, -0.073462508761, 0.515398670374, 1.099106630537, 0.68074534719, -0.086653615406, -0.202648655286, 0.010758611751, 0.044823623042, -0.000766690896, -0.004783458512};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 9:
            {
                data_size = 18;
                T data_local[18] = {0.001512487309, -0.000669141509, -0.014515578553, 0.012528896242, 0.087791251554, -0.02578644593, -0.270893783503, 0.049882830959, 0.873048407349, 1.015259790832, 0.337658923602, -0.077172161097, 0.000825140929, 0.042744433602, -0.016303351226, -0.018769396836, 0.000876502539, 0.001981193736};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 10:
            {
                data_size = 20;
                T data_local[20] = {0.001089170447, 0.00013524502, -0.01222064263, -0.002072363923, 0.064950924579, 0.016418869426, -0.225558972234, -0.100240215031, 0.667071338154, 1.0882515305, 0.542813011213, -0.050256540092, -0.045240772218, 0.07070356755, 0.008152816799, -0.028786231926, -0.001137535314, 0.006495728375, 0.000080661204, -0.000649589896};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            default:
            {
                break;
            }
            }
            break;
        }
        case vaidyanathan:
        {
            data_size = 24;
            T data_local[24] = {-0.000062906118, 0.000343631905, -0.00045395662, -0.000944897136, 0.002843834547, 0.000708137504, -0.008839103409, 0.003153847056, 0.01968721501, -0.014853448005, -0.035470398607, 0.038742619293, 0.055892523691, -0.077709750902, -0.083928884366, 0.131971661417, 0.135084227129, -0.194450471766, -0.263494802488, 0.201612161775, 0.635601059872, 0.572797793211, 0.250184129505, 0.045799334111};
            std::copy(data_local, data_local+data_size, data);
            break;
        }
        case battle:
        {
            switch(parameter)
            {
            case 1:
            {
                data_size = 23;
                T data_local[23] = {-0.0000867523, -0.000158601, 0.000361781, 0.000652922, -0.00155701, -0.00274588, 0.00706442, 0.012003, -0.0367309, -0.0488618, 0.280931, 0.578163, 0.280931, -0.0488618, -0.0367309, 0.012003, 0.00706442, -0.00274588, -0.00155701, 0.000652922, 0.000361781, -0.000158601, -0.000086752300000};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 3:
            {
                data_size = 41;
                T data_local[41] = {0.000103307, -0.000164264, -0.000201818, 0.000326749, 0.000395946, -0.00065562, -0.000780468, 0.00133086, 0.00154624, -0.00274529, -0.00307863, 0.00579932, 0.00614143, -0.0127154, -0.0121455, 0.0297468, 0.0226846, -0.0778079, -0.035498, 0.30683, 0.541736, 0.30683, -0.035498, -0.0778079, 0.0226846, 0.0297468, -0.0121455, -0.0127154, 0.00614143, 0.00579932, -0.00307863, -0.00274529, 0.00154624, 0.00133086, -0.000780468, -0.00065562, 0.000395946, 0.000326749, -0.000201818, -0.000164264, 0.000103307000000};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            case 5:
            {
                data_size = 59;
                T data_local[59] = {0.000101113, 0.000110709, -0.000159168, -0.000172685, 0.000251419, 0.000269842, -0.000398759, -0.000422485, 0.000635563, 0.000662836, -0.00101912, -0.00104207, 0.00164659, 0.00164132, -0.00268646, -0.00258816, 0.00444002, 0.00407882, -0.00746848, -0.00639886, 0.0128754, 0.00990635, -0.0229951, -0.0148537, 0.0433544, 0.0208414, -0.0914068, -0.0261771, 0.312869, 0.528374, 0.312869, -0.0261771, -0.0914068, 0.0208414, 0.0433544, -0.0148537, -0.0229951, 0.00990635, 0.0128754, -0.00639886, -0.00746848, 0.00407882, 0.00444002, -0.00258816, -0.00268646, 0.00164132, 0.00164659, -0.00104207, -0.00101912, 0.000662836, 0.000635563, -0.000422485, -0.000398759, 0.000269842, 0.000251419, -0.000172685, -0.000159168, 0.000110709, 0.0001011130};
                std::copy(data_local, data_local+data_size, data);
                break;
            }
            default:
            {
                break;
            }
            }
            break;
        }
        default:
        {
            break;
        }
        }

        if( filter_type == high )
            for( size_t i = 1; i < data_size; i += 2 )
                data[i] = -data[i];

        Matrix<T> result(data, data_size, data_size, 1);
        return result/result.Norm(two);
    }

    /** Forward Wavelet Transform (periodized, orthogonal)
     *  \brief Applies a periodized and orthogonal discrete wavelet transform.
     *  \param signal Signal to transform, must be length a power of 2.
     *  \param wcoef Result array, must be the same size as signal.
     *  \param column Column to transform
     *  \param coarsest_level Coarsest level of the wavelet transform
     *  \param intermediate Temporary array of size Height of signal
     *  \param intermediate_temp Temporary array of size Height of signal
     *  \author David Donoho <donoho@stat.stanford.edu> 1993
     *  \author Philippe Ganz <philippe.ganz@gmail.com> 2018
     */
    void FWT_PO(const Matrix<T>& signal,
                Matrix<T>& wcoef,
                unsigned int column,
                unsigned int coarsest_level,
                T* intermediate,
                T* intermediate_temp ) const
    {
        size_t level_max = (size_t) std::ceil(std::log2(signal.Height()));

#ifdef DO_ARGCHECKS
        if( (size_t) std::pow(2,level_max) != signal.Length() )
        {
            std::cerr << "Signal height must be length a power of two." << std::endl;
            throw;
        }

        if( coarsest_level >= level_max )
        {
            std::cerr << "The coarsest level must be in the [0, " << level_max << ") range." << std::endl;
            throw;
        }

        if( column >= signal.Width() )
        {
            std::cerr << "The column must be in the [0, " << signal.Width() << ") range." << std::endl;
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for( size_t i = 0; i < signal.Height(); ++i )
            intermediate[i] = signal[i*wcoef.Width() + column];

        for( size_t level = level_max, level_size = signal.Height(); level > coarsest_level; --level, level_size /= 2 )
        {
            #pragma omp parallel for
            for( size_t pass_index = 0; pass_index < level_size/2; ++pass_index )
            {
                T low_pass_local_coef = 0.0;
                size_t low_pass_offset = 2*pass_index;
                T high_pass_local_coef = 0.0;
                int high_pass_offset = 2*pass_index+1;

                for( size_t filter_index = 0; filter_index < low_pass_filter_.Length(); ++filter_index )
                {
                    low_pass_local_coef += low_pass_filter_[filter_index] * intermediate[low_pass_offset];

                    ++low_pass_offset;
                    if( low_pass_offset >= level_size )
                        low_pass_offset -= level_size;

                    high_pass_local_coef += high_pass_filter_[filter_index] * intermediate[high_pass_offset];

                    --high_pass_offset;
                    if( high_pass_offset < 0 )
                        high_pass_offset += level_size;
                }

                intermediate_temp[pass_index] = low_pass_local_coef;
                intermediate_temp[pass_index + level_size/2] = high_pass_local_coef;
            }

            #pragma omp parallel for simd
            for( size_t i = 0; i < level_size; ++i )
                intermediate[i] = intermediate_temp[i];
        }

        #pragma omp parallel for simd
        for( size_t i = 0; i < signal.Height(); ++i )
            wcoef[i*wcoef.Width() + column] = intermediate[i];
    }

    /** Inverse Wavelet Transform (periodized, orthogonal)
     *  \brief Applies a periodized and orthogonal inverse discrete wavelet transform.
     *  \param wcoef Wavelet coefficients to transform back, must be length a power of 2.
     *  \param signal Result array, must be the same size as wcoef.
     *  \param column Column to transform, -1 to transform all
     *  \param coarsest_level Coarsest level of the wavelet transform
     *  \param intermediate Temporary array of size Height of signal
     *  \param intermediate_temp Temporary array of size Height of signal
     *  \author David Donoho <donoho@stat.stanford.edu> 1993
     *  \author Philippe Ganz <philippe.ganz@gmail.com> 2018
     */
    void IWT_PO(const Matrix<T>& wcoef,
                Matrix<T>& signal,
                unsigned int column,
                unsigned int coarsest_level,
                T* intermediate,
                T* intermediate_temp ) const
    {
        size_t level_max = (size_t) std::ceil(std::log2(signal.Height()));

#ifdef DO_ARGCHECKS
        if( (size_t) std::pow(2,level_max) != signal.Length() )
        {
            std::cerr << "Signal height must be length a power of two." << std::endl;
            throw;
        }

        if( coarsest_level >= level_max )
        {
            std::cerr << "The coarsest level must be in the [, " << level_max << ") range." << std::endl;
            throw;
        }

        if( column >= signal.Width() )
        {
            std::cerr << "The column must be in the [, " << signal.Width() << ") range." << std::endl;
            throw;
        }
#endif // DO_ARGCHECKS

        #pragma omp parallel for simd
        for( size_t i = 0; i < (size_t) std::pow(2, coarsest_level); ++i )
            intermediate[i] = wcoef[i*wcoef.Width() + column];

        for( size_t level = (size_t) std::pow(2, coarsest_level), level_size = 1; level <= level_max; ++level, level_size *= 2 )
        {
            #pragma omp parallel for
            for( size_t pass_index = 0; pass_index < level_size; ++pass_index )
            {
                T even_local_coef = 0.0;
                int low_pass_offset = pass_index;
                T odd_local_coef = 0.0;
                size_t high_pass_offset = pass_index;

                for( size_t filter_index = 0; filter_index < (low_pass_filter_.Length() + 1) / 2; ++filter_index )
                {
                    even_local_coef += low_pass_filter_[2*filter_index] * intermediate[low_pass_offset];

                    --low_pass_offset;
                    if( low_pass_offset < 0 )
                        low_pass_offset += level_size;

                    odd_local_coef += high_pass_filter_[2*filter_index] * wcoef[(level_size + high_pass_offset)*wcoef.Width() + column];

                    ++high_pass_offset;
                    if( high_pass_offset >= level_size )
                        high_pass_offset -= level_size;
                }

                low_pass_offset = pass_index;
                high_pass_offset = pass_index;
                for( size_t filter_index = 0; filter_index < low_pass_filter_.Length() / 2; ++filter_index )
                {
                    odd_local_coef += low_pass_filter_[2*filter_index+1] * intermediate[low_pass_offset];

                    --low_pass_offset;
                    if( low_pass_offset < 0 )
                        low_pass_offset += level_size;

                    even_local_coef += high_pass_filter_[2*filter_index+1] * wcoef[(level_size + high_pass_offset)*wcoef.Width() + column];

                    ++high_pass_offset;
                    if( high_pass_offset >= level_size )
                        high_pass_offset -= level_size;
                }

                intermediate_temp[2*pass_index] = even_local_coef;
                intermediate_temp[2*pass_index + 1] = odd_local_coef;
            }

            #pragma omp parallel for simd
            for( size_t i = 0; i < 2*level_size; ++i )
                intermediate[i] = intermediate_temp[i];
        }

        #pragma omp parallel for simd
        for( size_t i = 0; i < signal.Height(); ++i )
            signal[i*signal.Width() + column] = intermediate[i];
    }
};

} // namespace alias

#endif // ASTROQUT_UTILS_OPERATOR_WAVELET_HPP
