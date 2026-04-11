/*
 * owLeakyReLUActivation.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owActivation.hpp"
#include "../core/owSimd.hpp"

namespace ow {

/**
 * @class owLeakyReLUActivation
 * @brief Implements the Leaky Rectified Linear Unit (LeakyReLU) activation function.
 * 
 * LeakyReLU is a variation of ReLU that allows a small, non-zero gradient when the unit
 * is not active (input < 0). This prevents the "dying ReLU" problem where neurons 
 * stop learning entirely because their gradient is zero.
 * 
 * In mobile and real-time industrial applications, LeakyReLU provides a robust balance
 * between computational efficiency (no complex transcendental functions like exp) 
 * and model stability during deep training.
 */
class owLeakyReLUActivation : public owActivation {
public:
    /**
     * @brief Constructs a LeakyReLU activation.
     * @param alpha The slope for negative inputs (default: 0.01).
     */
    owLeakyReLUActivation(float alpha = 0.01f) : m_alpha(alpha) {}

    /**
     * @brief Forward pass: f(x) = x if x > 0, else alpha * x.
     * @param input Input tensor.
     * @return Output tensor with LeakyReLU applied.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> out = input;
        float* data = out.data();
        size_t n = out.size();

        #ifdef __AVX2__
        __m256 v_alpha = _mm256_set1_ps(m_alpha);
        __m256 v_zero = _mm256_setzero_ps();
        for (size_t i = 0; i <= n - 8; i += 8) {
            __m256 v_x = _mm256_loadu_ps(data + i);
            __m256 v_mask = _mm256_cmp_ps(v_x, v_zero, _CMP_GT_OQ);
            __m256 v_leaky = _mm256_mul_ps(v_x, v_alpha);
            _mm256_storeu_ps(data + i, _mm256_blendv_ps(v_leaky, v_x, v_mask));
        }
        for (size_t i = (n / 8) * 8; i < n; ++i) data[i] = (data[i] > 0.0f ? data[i] : m_alpha * data[i]);
        #elif defined(OW_ARM_NEON)
        float32x4_t v_alpha = vdupq_n_f32(m_alpha);
        float32x4_t v_zero = vdupq_n_f32(0.0f);
        for (size_t i = 0; i <= n - 4; i += 4) {
            float32x4_t v_x = vld1q_f32(data + i);
            uint32x4_t v_mask = vcgtq_f32(v_x, v_zero);
            float32x4_t v_leaky = vmulq_f32(v_x, v_alpha);
            vst1q_f32(data + i, vbslq_f32(v_mask, v_x, v_leaky));
        }
        for (size_t i = (n / 4) * 4; i < n; ++i) data[i] = (data[i] > 0.0f ? data[i] : m_alpha * data[i]);
        #else
        for (size_t i = 0; i < n; ++i) data[i] = (data[i] > 0.0f ? data[i] : m_alpha * data[i]);
        #endif
        return out;
    }

    /**
     * @brief Backward pass: gradient is 1.0 if x > 0, else alpha.
     * @param input Original input tensor.
     * @param outputGradient Gradient from the subsequent layer.
     * @return Resulting gradient for backpropagation.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& input, const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> grad = outputGradient;
        float* gData = grad.data();
        const float* iData = input.data();
        size_t n = grad.size();

        #ifdef __AVX2__
        __m256 v_alpha = _mm256_set1_ps(m_alpha);
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_zero = _mm256_setzero_ps();
        for (size_t i = 0; i <= n - 8; i += 8) {
            __m256 v_x = _mm256_loadu_ps(iData + i);
            __m256 v_mask = _mm256_cmp_ps(v_x, v_zero, _CMP_GT_OQ);
            __m256 v_mult = _mm256_blendv_ps(v_alpha, v_one, v_mask);
            _mm256_storeu_ps(gData + i, _mm256_mul_ps(_mm256_loadu_ps(gData + i), v_mult));
        }
        for (size_t i = (n / 8) * 8; i < n; ++i) gData[i] *= (iData[i] > 0.0f ? 1.0f : m_alpha);
        #elif defined(OW_ARM_NEON)
        float32x4_t v_alpha = vdupq_n_f32(m_alpha);
        float32x4_t v_one = vdupq_n_f32(1.0f);
        float32x4_t v_zero = vdupq_n_f32(0.0f);
        for (size_t i = 0; i <= n - 4; i += 4) {
            float32x4_t v_x = vld1q_f32(iData + i);
            uint32x4_t v_mask = vcgtq_f32(v_x, v_zero);
            float32x4_t v_mult = vbslq_f32(v_mask, v_one, v_alpha);
            vst1q_f32(gData + i, vmulq_f32(vld1q_f32(gData + i), v_mult));
        }
        for (size_t i = (n / 4) * 4; i < n; ++i) gData[i] *= (iData[i] > 0.0f ? 1.0f : m_alpha);
        #else
        for (size_t i = 0; i < n; ++i) gData[i] *= (iData[i] > 0.0f ? 1.0f : m_alpha);
        #endif
        return grad;
    }

    /**
     * @brief Deep copy of the LeakyReLU instance.
     * @return Shared pointer to new instance.
     */
    std::shared_ptr<owActivation> clone() const override { return std::make_shared<owLeakyReLUActivation>(m_alpha); }

private:
    float m_alpha; ///< Leakage coefficient for negative inputs.
};

} // namespace ow
