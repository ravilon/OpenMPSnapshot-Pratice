/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"
#include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::trait
{
    //! The CUDA/HIP RT device memory buffer type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct BufType<DevUniformCudaHipRt<TApi>, TElem, TDim, TIdx>
    {
        using type = BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>;
    };

    //! The BufUniformCudaHipRt device type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct DevType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = DevUniformCudaHipRt<TApi>;
    };

    //! The BufUniformCudaHipRt device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetDev<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> DevUniformCudaHipRt<TApi>
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The BufUniformCudaHipRt dimension getter trait.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct DimType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The BufUniformCudaHipRt memory element type get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct ElemType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The BufUniformCudaHipRt width get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetExtents<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The BufUniformCudaHipRt native pointer get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>& buf) -> TElem*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The BufUniformCudaHipRt pointer on device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf,
            DevUniformCudaHipRt<TApi> const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spBufImpl->m_pMem;
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>& buf,
            DevUniformCudaHipRt<TApi> const& dev) -> TElem*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spBufImpl->m_pMem;
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPitchesInBytes<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf) const
            -> Vec<TDim, TIdx>
        {
            return GetPitchesInBytes<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>{}(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufUniformCudaHipRt offset get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& /*buf*/) const
            -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The BufUniformCudaHipRt idx type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct IdxType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on CUDA/HIP device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevUniformCudaHipRt<TApi> const&)
            -> TElem const*
        {
            // TODO: Check if the memory is mapped at all!
            TElem* pDev(nullptr);

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostGetDevicePointer(
                &pDev,
                const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                0));

            return pDev;
        }

        ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevUniformCudaHipRt<TApi> const&)
            -> TElem*
        {
            // TODO: Check if the memory is mapped at all!
            TElem* pDev(nullptr);

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostGetDevicePointer(&pDev, getPtrNative(buf), 0));

            return pDev;
        }
    };

    //! The MakeConstBuf trait for CUDA/HIP buffers.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
        {
            return ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>(buf);
        }

        ALPAKA_FN_HOST static auto makeConstBuf(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>&& buf)
            -> ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
        {
            return ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>(std::move(buf));
        }
    };

    //! The CUDA/HIP memory allocation trait specialization.
    template<typename TApi, typename TElem, typename Dim, typename TIdx>
    struct BufAlloc<TElem, Dim, TIdx, DevUniformCudaHipRt<TApi>>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocBuf(DevUniformCudaHipRt<TApi> const& dev, TExtent const& extent)
            -> BufUniformCudaHipRt<TApi, TElem, Dim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

            void* memPtr = nullptr;
            std::size_t rowPitchInBytes = 0u;
            if(getExtentProduct(extent) != 0)
            {
                if constexpr(Dim::value == 0)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::malloc(&memPtr, sizeof(TElem)));
                }
                else if constexpr(Dim::value == 1)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        TApi::malloc(&memPtr, static_cast<std::size_t>(getWidth(extent)) * sizeof(TElem)));
                }
                else if constexpr(Dim::value == 2)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::mallocPitch(
                        &memPtr,
                        &rowPitchInBytes,
                        static_cast<std::size_t>(getWidth(extent)) * sizeof(TElem),
                        static_cast<std::size_t>(getHeight(extent))));
                }
                else if constexpr(Dim::value == 3)
                {
                    typename TApi::Extent_t const extentVal = TApi::makeExtent(
                        static_cast<std::size_t>(getWidth(extent)) * sizeof(TElem),
                        static_cast<std::size_t>(getHeight(extent)),
                        static_cast<std::size_t>(getDepth(extent)));
                    typename TApi::PitchedPtr_t pitchedPtrVal;
                    pitchedPtrVal.ptr = nullptr;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::malloc3D(&pitchedPtrVal, extentVal));
                    memPtr = pitchedPtrVal.ptr;
                    rowPitchInBytes = pitchedPtrVal.pitch;
                }
            }
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__;
            if constexpr(Dim::value >= 1)
                std::cout << " ew: " << getWidth(extent);
            if constexpr(Dim::value >= 2)
                std::cout << " eh: " << getHeight(extent);
            if constexpr(Dim::value >= 3)
                std::cout << " ed: " << getDepth(extent);
            std::cout << " ptr: " << memPtr;
            if constexpr(Dim::value >= 2)
                std::cout << " rowpitch: " << rowPitchInBytes;
            std::cout << std::endl;
#    endif
            return {
                dev,
                reinterpret_cast<TElem*>(memPtr),
                [](TElem* ptr) { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::free(ptr)); },
                extent,
                rowPitchInBytes};
        }
    };

    //! The CUDA/HIP stream-ordered memory allocation capability trait specialization.
    template<typename TApi, typename TDim>
    struct HasAsyncBufSupport<TDim, DevUniformCudaHipRt<TApi>> : std::bool_constant<TDim::value <= 1>
    {
    };

    //! The CUDA/HIP stream-ordered memory allocation trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct AsyncBufAlloc<TElem, TDim, TIdx, DevUniformCudaHipRt<TApi>>
    {
        static_assert(
            TDim::value <= 1,
            "CUDA/HIP devices support only one-dimensional stream-ordered memory buffers.");

        template<typename TQueue, typename TExtent>
        ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, [[maybe_unused]] TExtent const& extent)
            -> BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(TDim::value == Dim<TExtent>::value, "extent must have the same dimension as the buffer");
            auto const width = getExtentProduct(extent); // handles 1D and 0D buffers

            auto const& dev = getDev(queue);
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));
            void* memPtr = nullptr;
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                TApi::mallocAsync(&memPtr, static_cast<std::size_t>(width) * sizeof(TElem), queue.getNativeHandle()));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " ew: " << width << " ptr: " << memPtr << std::endl;
#    endif
            return {
                dev,
                reinterpret_cast<TElem*>(memPtr),
                [q = std::move(queue)](TElem* ptr)
                { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::freeAsync(ptr, q.getNativeHandle())); },
                extent,
                static_cast<std::size_t>(width) * sizeof(TElem)};
        }
    };

    //! The pinned/mapped memory allocation capability trait specialization.
    template<typename TApi>
    struct HasMappedBufSupport<PlatformUniformCudaHipRt<TApi>> : public std::true_type
    {
    };

    //! The pinned/mapped memory allocation trait specialization for the CUDA/HIP devices.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped<PlatformUniformCudaHipRt<TApi>, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocMappedBuf(
            DevCpu const& host,
            PlatformUniformCudaHipRt<TApi> const& /*platform*/,
            TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Allocate CUDA/HIP page-locked memory on the host, mapped into the CUDA/HIP address space and
            // accessible to all CUDA/HIP devices.
            TElem* memPtr = nullptr;
            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostMalloc(
                reinterpret_cast<void**>(&memPtr),
                sizeof(TElem) * static_cast<std::size_t>(getExtentProduct(extent)),
                TApi::hostMallocMapped | TApi::hostMallocPortable));
            auto deleter = [](TElem* ptr) { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::hostFree(ptr)); };

            return BufCpu<TElem, TDim, TIdx>(host, memPtr, std::move(deleter), extent);
        }
    };

} // namespace alpaka::trait

#endif
