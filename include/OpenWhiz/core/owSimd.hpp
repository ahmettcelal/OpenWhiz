/*
 * owSimd.hpp
 *
 *  Created on: Apr 10, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define OW_X86_SIMD
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define OW_ARM_NEON
#endif

namespace ow {
namespace simd {

#ifdef OW_X86_SIMD
    // AVX2 / AVX-512 checks are usually done via macros like __AVX2__, __AVX512F__
#endif

#ifdef OW_ARM_NEON
    // ARM Neon is generally available on aarch64
#endif

} // namespace simd
} // namespace ow
