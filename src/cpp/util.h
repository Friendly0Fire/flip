#include <numbers>

#pragma once

#define FWD(...) std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

#define LIFT(X) [](auto &&... args) \
    noexcept(noexcept(X(FWD(args)...)))  \
    -> decltype(X(FWD(args)...)) \
{  \
    return X(FWD(args)...); \
}

#define DEFAULT_ILLUMINANT floatN<3>{ 0.950428545f, 1.000000000f, 1.088900371f }
#define INV_DEFAULT_ILLUMINANT floatN<3>{ 1.052156925f, 1.000000000f, 0.918357670f }

namespace FLIP {

constexpr float PI = std::numbers::pi_v<float>;

static constexpr float ToneMappingCoefficients[3][6] =
{
    { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },                                                 // Reinhard.
    { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  // ACES, 0.6 is pre-exposure cancellation.
    { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },                                    // Hable.
};

constexpr inline float square(float x) {
    return x * x;
}

//  Pixels per degree (PPD).
constexpr inline float calculatePPD(const float dist, const float resolutionX, const float monitorWidth) {
    return dist * (resolutionX / monitorWidth) * (PI / 180.0f);
}

constexpr inline void solveSecondDegree(float& xMin, float& xMax, float a, float b, float c) {
    //  a * x^2 + b * x + c = 0
    if(a == 0.0f) {
        xMin = xMax = -c / b;
        return;
    }

    float d1 = -0.5f * (b / a);
    float d2 = std::sqrt((d1 * d1) - (c / a));
    xMin = d1 - d2;
    xMax = d1 + d2;
}

inline float sRGBToLinearRGB(float sC) {
    if(sC <= 0.04045f) {
        return sC / 12.92f;
    }
    return std::pow((sC + 0.055f) / 1.055f, 2.4f);
}

inline float LinearRGBTosRGB(float lC) {
    if(lC <= 0.0031308f) {
        return lC * 12.92f;
    }

    return 1.055f * std::pow(lC, 1.0f / 2.4f) - 0.055f;
}

}
