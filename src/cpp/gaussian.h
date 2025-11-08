#pragma once

#include "util.h"
#include "vecs.h"

namespace FLIP {

static const struct xGaussianConstants {
    xGaussianConstants() = default;
    float3 a1 = { 1.0f, 1.0f, 34.1f };
    float3 b1 = { 0.0047f, 0.0053f, 0.04f };
    float3 a2 = { 0.0f, 0.0f, 13.5f };
    float3 b2 = { 1.0e-5f, 1.0e-5f, 0.025f };
} GaussianConstants;  // Constants for Gaussians -- see paper for details.

inline float Gaussian(const float x, const float sigma) // 1D Gaussian (without normalization factor).
{
    return std::exp(-(x * x) / (2.0f * sigma * sigma));
}

inline float Gaussian(const float x2, const float a, const float b) // 1D Gaussian in alternative form (see FLIP paper).
{
    constexpr float pi = Pi;
    constexpr float pi_sq = Pi * Pi;
    return a * std::sqrt(pi / b) * std::exp(-pi_sq * x2 / b);
}

// This function is needed to separate sum of Gaussians filters See separatedConvolutions.pdf in the FLIP repository:
// https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
static inline float GaussianSqrt(const float x2, const float a, const float b) {
    const float pi = float(Pi);
    const float pi_sq = float(Pi * Pi);
    return std::sqrt(a * std::sqrt(pi / b)) * std::exp(-pi_sq * x2 / b);
}

static int calculateSpatialFilterRadius(const float ppd) {
    const float pi_sq = float(Pi * Pi);

    float maxScaleParameter = std::max(std::max(std::max(GaussianConstants.b1.x(), GaussianConstants.b1.y()), std::max(GaussianConstants.b1.z(), GaussianConstants.b2.x())), std::max(GaussianConstants.b2.y(), GaussianConstants.b2.z()));
    int radius = int(std::ceil(3.0f * std::sqrt(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter.

    return radius;
}

}
