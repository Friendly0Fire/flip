#pragma once

#include "util.h"
#include "vecs.h"

namespace FLIP {

inline static constexpr struct xGaussianConstants {
    float3 a1 { 1.0f, 1.0f, 34.1f };
    float3 b1 { 0.0047f, 0.0053f, 0.04f };
    float3 a2 { 0.0f, 0.0f, 13.5f };
    float3 b2 { 1.0e-5f, 1.0e-5f, 0.025f };
} GaussianConstants;  // Constants for Gaussians -- see paper for details.

// 1D Gaussian (without normalization factor).
inline float Gaussian(const float x, const float sigma) {
    return std::exp(-(x * x) / (2.0f * sigma * sigma));
}

// 1D Gaussian in alternative form (see FLIP paper).
inline float Gaussian(const float x2, const float a, const float b) {
    constexpr float pi = Pi;
    constexpr float pi_sq = Pi * Pi;
    return a * std::sqrt(pi / b) * std::exp(-pi_sq * x2 / b);
}

// This function is needed to separate sum of Gaussians filters See separatedConvolutions.pdf in the FLIP repository:
// https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
inline float GaussianSqrt(const float x2, const float a, const float b) {
    constexpr float pi = float(Pi);
    constexpr float pi_sq = float(Pi * Pi);
    return std::sqrt(a * std::sqrt(pi / b)) * std::exp(-pi_sq * x2 / b);
}

inline int calculateSpatialFilterRadius(const float ppd) {
    constexpr float pi_sq = float(Pi * Pi);
    constexpr float max_scale_parameter = std::max( { GaussianConstants.b1.x(), GaussianConstants.b1.y(), GaussianConstants.b1.z(), GaussianConstants.b2.x(), GaussianConstants.b2.y(), GaussianConstants.b2.z() } );
    int radius = int(std::ceil(3.0f * std::sqrt(max_scale_parameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter.

    return radius;
}

}
