#ifndef PRAGMA
#define PRAGMA

#define TOSTRING(a) #a

#if defined(__clang__)
    // # pragma clang loop unroll(n)
    #define UNROLL(n) \
    _Pragma( TOSTRING(clang loop unroll(full)))
    #define UNROLL_S(n) \
    _Pragma( TOSTRING(clang loop unroll(full)))
#elif defined (__FUJITSU)
    // #pragma loop fullunroll_pre_simd n
    #define UNROLL(n) \
    _Pragma( TOSTRING(loop fullunroll_pre_simd n))
    #define UNROLL_S(n) \
    _Pragma( TOSTRING(loop fullunroll_pre_simd n))
#elif defined(__GNUC__)
    // #pragma GCC unroll (n)
	#define UNROLL(n) \
    _Pragma( TOSTRING(GCC unroll (n)))
	#define UNROLL_S(n) \
    _Pragma( TOSTRING(GCC unroll (n)))
#else
    //#pragma unroll(5)
    #define UNROLL(n) \
    _Pragma( TOSTRING( pragma unroll (n)))
    #define UNROLL_S(n)
#endif

#if defined ( SMILEI_ACCELERATOR_GPU_OMP )
    #define ATOMIC(mode) \
    _Pragma( TOSTRING(omp atomic mode))
#elif defined ( SMILEI_ACCELERATOR_GPU_OACC )
    #define ATOMIC(mode) \
    _Pragma( TOSTRING(acc atomic mode))
#endif

#endif
