/*
 * owTanhActivation.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owActivation.hpp"
#include "../core/owSimd.hpp"

namespace ow {

/**
 * @class owTanhActivation
 * @brief Hyperbolic Tangent (tanh) activation function.
 *
 * The tanh activation function maps input values to a range between -1 and 1.
 * It is mathematically defined as:
 * f(x) = tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
 *
 * Compared to the Sigmoid function, tanh is zero-centered, which often leads to 
 * faster convergence during training in deep neural networks. Its output mean 
 * is closer to zero, which helps in centering the data for the subsequent layers.
 *
 * @note In industrial and embedded systems, tanh provides a smooth, continuous 
 * transition which can be beneficial for control systems and signal processing 
 * where abrupt changes (like in ReLU) might cause instability.
 * 
 * @note For mobile and web applications, tanh is computationally more expensive 
 * than ReLU but provides better performance in Recurrent Neural Networks (RNNs) 
 * and LSTMs where it is the standard activation for state transitions.
 */
class owTanhActivation : public owActivation {
public:
    /**
     * @brief Performs the forward pass using the tanh function.
     * 
     * Applies the hyperbolic tangent f(x) = tanh(x) element-wise to the input tensor.
     * This function is crucial for modeling non-linear relationships in data.
     *
     * @param input The input tensor containing weighted sums from a linear layer.
     * @return owTensor<float, 2> The activated output tensor with values in the range (-1, 1).
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> out = input;
        for (size_t i = 0; i < out.size(); ++i) out.data()[i] = std::tanh(out.data()[i]);
        return out;
    }

    /**
     * @brief Performs the backward pass to calculate gradients.
     * 
     * Computes the gradient of the loss with respect to the input by applying 
     * the chain rule with the derivative of tanh:
     * f'(x) = 1 - tanh(x)^2
     *
     * This implementation is optimized with AVX2 SIMD instructions where available, 
     * providing significant performance gains in computer-based high-performance 
     * computing (HPC) environments. For mobile and web architectures, it falls 
     * back to a standard loop.
     *
     * @param input The original input tensor from the forward pass.
     * @param outputGradient The gradient of the loss with respect to the output.
     * @return owTensor<float, 2> The gradient with respect to the input, calculated as 
     * outputGradient * (1 - tanh(input)^2).
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& input, const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> grad = outputGradient;
        float* gData = grad.data();
        const float* iData = input.data();
        size_t n = grad.size();

        #ifdef __AVX2__
        __m256 v_one = _mm256_set1_ps(1.0f);
        for (size_t i = 0; i <= n - 8; i += 8) {
            float t[8];
            for(int j=0; j<8; ++j) t[j] = std::tanh(iData[i+j]);
            __m256 v_t = _mm256_loadu_ps(t);
            __m256 v_grad = _mm256_loadu_ps(gData + i);
            _mm256_storeu_ps(gData + i, _mm256_mul_ps(v_grad, _mm256_sub_ps(v_one, _mm256_mul_ps(v_t, v_t))));
        }
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            float t = std::tanh(iData[i]);
            gData[i] *= (1.0f - t * t);
        }
        #elif defined(OW_ARM_NEON)
        float32x4_t v_one = vdupq_n_f32(1.0f);
        for (size_t i = 0; i <= n - 4; i += 4) {
            float t[4];
            for(int j=0; j<4; ++j) t[j] = std::tanh(iData[i+j]);
            float32x4_t v_t = vld1q_f32(t);
            float32x4_t v_grad = vld1q_f32(gData + i);
            vst1q_f32(gData + i, vmulq_f32(v_grad, vsubq_f32(v_one, vmulq_f32(v_t, v_t))));
        }
        for (size_t i = (n / 4) * 4; i < n; ++i) {
            float t = std::tanh(iData[i]);
            gData[i] *= (1.0f - t * t);
        }
        #else
        for (size_t i = 0; i < n; ++i) {
            float t = std::tanh(iData[i]);
            gData[i] *= (1.0f - t * t);
        }
        #endif
        return grad;
    }

    /**
     * @brief Creates a deep copy of the tanh activation object.
     * @return std::shared_ptr<owActivation> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owActivation> clone() const override { return std::make_shared<owTanhActivation>(); }
};

} // namespace ow
