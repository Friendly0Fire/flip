#pragma once

#include <cmath>
#include <numbers>
#include <string_view>
#include <format>

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

constexpr float Pi = std::numbers::pi_v<float>;

static constexpr float ToneMappingCoefficients[3][6] =
{
    { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f },                                                 // Reinhard.
    { 0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f },  // ACES, 0.6 is pre-exposure cancellation.
    { 0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f },                                    // Hable.
};

enum class tonemapper {
    reinhard = 0,
    aces = 1,
    hable = 2
};

inline std::string_view to_string(tonemapper tm) {
    switch(tm) {
    case tonemapper::reinhard:
        return "Reinhard";
    case tonemapper::aces:
        return "ACES";
    case tonemapper::hable:
        return "Hable";
    }

    return "Unknown";
}

constexpr inline float square(float x) {
    return x * x;
}

//  Pixels per degree (PPD).
constexpr inline float calculate_ppd(const float dist, const float resolutionX, const float monitorWidth) {
    return dist * (resolutionX / monitorWidth) * (Pi / 180.0f);
}

struct second_degree_solution {
    float min, max;
};
constexpr inline second_degree_solution solve_second_degree(float& xMin, float& xMax, float a, float b, float c) {
    //  a * x^2 + b * x + c = 0
    if(a == 0.0f)
        return { -c / b, -c / b };

    float aInv = 1.f / a;
    float d1 = -0.5f * b * aInv;
    float d2 = std::sqrt(square(d1) - c * aInv);
    return { d1 - d2, d1 + d2 };
}

inline float srgb_to_linear_rgb(float sC) {
    if(sC <= 0.04045f) {
        return sC / 12.92f;
    }
    return std::pow((sC + 0.055f) / 1.055f, 2.4f);
}

inline float linear_rgb_to_srgb(float lC) {
    if(lC <= 0.0031308f) {
        return lC * 12.92f;
    }

    return 1.055f * std::pow(lC, 1.0f / 2.4f) - 0.055f;
}

struct exposure_range {
    float min = std::numeric_limits<float>::infinity(), max = std::numeric_limits<float>::infinity();

    bool unbounded() const {
        return std::isinf(min) || std::isinf(max);
    }

    float range() const {
        return max - min;
    }
};

}

template<>
struct std::formatter<FLIP::tonemapper> : std::formatter<std::string_view> {
    using parent_t = std::formatter<std::string_view>;

    template<class FmtContext>
    typename FmtContext::iterator format(FLIP::tonemapper tm, FmtContext& ctx) const {
        switch(tm) {
        case FLIP::tonemapper::reinhard:
            return parent_t::format("Reinhard", ctx);
        case FLIP::tonemapper::aces:
            return parent_t::format("ACES", ctx);
        case FLIP::tonemapper::hable:
            return parent_t::format("Hable", ctx);
        }
        return parent_t::format("<unknown>", ctx);
    }
};

template<>
struct std::formatter<FLIP::exposure_range> {
    using parent_t = std::formatter<std::string_view>;
    template<class FmtContext>
    FmtContext::iterator format(FLIP::exposure_range exp, FmtContext& ctx) const {
        return std::format_to(ctx.out(), "{}<=>{}", exp.min, exp.max);
    }
};