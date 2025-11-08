#pragma once

#include <algorithm>
#include <span>
#include <ranges>

#include "util.h"

namespace FLIP {

struct int3 {
    std::array<int, 3> data = { 0 };

    inline int x() const { return data[0]; }
    inline int y() const { return data[1]; }
    inline int z() const { return data[2]; }

    friend bool operator==(const int3& a, const int3& b) {
        return a.data == b.data;
    }

    constexpr int& operator[](size_t i) {
        return data[i];
    }

    constexpr int operator[](size_t i) const {
        return data[i];
    }
};

template<size_t N>
class floatN;

template<typename T, size_t N>
concept floatN_like = std::same_as<std::decay_t<T>, floatN<N>> || std::convertible_to<T, float>;
static_assert(floatN_like<floatN<3>, 3>);
static_assert(floatN_like<float, 3>);
static_assert(floatN_like<int, 3>);
static_assert(!floatN_like<floatN<2>, 3>);

template<size_t N>
class floatN {
    constexpr static float& at(floatN& f, size_t i) {
        return f.data[i];
    }
    constexpr static const float& at(const floatN& f, size_t i) {
        return f.data[i];
    }

    constexpr static float at(float f, size_t) {
        return f;
    }
public:
    inline static constexpr size_t count = N;

    std::array<float, N> data = { 0.f };

    auto&& x(this auto&& self)                   { return FWD(self).data[0]; }
    auto&& y(this auto&& self) requires (N >= 2) { return FWD(self).data[1]; }
    auto&& z(this auto&& self) requires (N >= 3) { return FWD(self).data[2]; }
    auto&& w(this auto&& self) requires (N >= 4) { return FWD(self).data[3]; }

    auto&& r(this auto&& self)                   { return FWD(self).data[0]; }
    auto&& g(this auto&& self) requires (N >= 2) { return FWD(self).data[1]; }
    auto&& b(this auto&& self) requires (N >= 3) { return FWD(self).data[2]; }
    auto&& a(this auto&& self) requires (N >= 4) { return FWD(self).data[3]; }

    constexpr floatN() = default;

    constexpr floatN(float v) {
        std::ranges::fill(data, v);
    }

    constexpr floatN(const float* pColor) {
        std::ranges::copy_n(pColor, N, data.begin());
    }

    constexpr floatN(const unsigned char* pColor) {
        std::span color(pColor, N);
        std::ranges::copy(color
            | std::views::transform(
                [](unsigned char c) { return float(c) / 255.f; }), data);
    }

    template<typename... Args> requires (sizeof...(Args) <= N && sizeof...(Args) > 1 && (std::is_arithmetic_v<Args> && ...))
    constexpr floatN(Args... args) {
        size_t i = 0;
        ((data[i++] = args), ...);
    }

    constexpr float& operator[](size_t i) {
        return data[i];
    }

    constexpr float operator[](size_t i) const {
        return data[i];
    }

    template<size_t N2> requires (N != N2)
    constexpr floatN& operator=(const floatN<N2>& v) {
        std::ranges::copy_n(v.data.begin(), std::min(N, N2), data.begin());
        return *this;
    }
    constexpr floatN& operator=(const floatN& v) {
        std::ranges::copy_n(v.data.begin(), N, data.begin());
        return *this;
    }
    constexpr floatN& operator=(floatN&& v) = default;

    template<size_t N2> requires (N != N2)
    constexpr floatN(const floatN<N2>& v) {
        *this = v;
    }
    constexpr floatN(const floatN& v) {
        *this = v;
    }
    constexpr floatN(floatN&&) = default;

    constexpr bool operator==(const floatN& c) const = default;
    constexpr bool operator!=(const floatN& c) const = default;

    template<floatN_like<N> T>
    constexpr floatN& operator+=(const T& c) {
        for(size_t i = 0; i < N; ++i)
            data[i] += at(c, i);
        return *this;
    }

    template<floatN_like<N> T>
    constexpr friend floatN operator+(const floatN& a, const T& b) {
        floatN c = a;
        c += b;
        return c;
    }

    constexpr friend floatN operator+(float a, const floatN& b) {
        return operator+(b, a);
    }

    template<floatN_like<N> T>
    constexpr floatN& operator-=(const T& c) {
        for(size_t i = 0; i < N; ++i)
            data[i] -= at(c, i);
        return *this;
    }

    template<floatN_like<N> T>
    constexpr friend floatN operator-(const floatN& a, const T& b) {
        floatN c = a;
        c -= b;
        return c;
    }

    constexpr friend floatN operator-(float a, const floatN& b) {
        return operator-(floatN(a), b);
    }

    template<floatN_like<N> T>
    constexpr floatN& operator*=(const T& c) {
        for(size_t i = 0; i < N; ++i)
            data[i] *= at(c, i);
        return *this;
    }

    template<floatN_like<N> T>
    constexpr friend floatN operator*(const floatN& a, const T& b) {
        floatN c = a;
        c *= b;
        return c;
    }

    constexpr friend floatN operator*(float a, const floatN& b) {
        return operator*(b, a);
    }

    template<floatN_like<N> T>
    constexpr floatN& operator/=(const T& c) {
        for(size_t i = 0; i < N; ++i)
            data[i] /= at(c, i);
        return *this;
    }

    template<floatN_like<N> T>
    constexpr friend floatN operator/(const floatN& a, const T& b) {
        floatN c = a;
        c /= b;
        return c;
    }

    constexpr friend floatN operator/(float a, const floatN& b) {
        return operator/(floatN(a), b);
    }

    template<size_t N2 = N>
    constexpr friend floatN spread(const floatN& a, const floatN& b, auto&& callable) {
        floatN c;
        for(size_t i = 0; i < N2; ++i)
            c[i] = callable(a[i], b[i]);
        return c;
    }

    template<size_t N2 = N>
    constexpr floatN spread(auto&& callable) const {
        floatN v = *this;
        for(size_t i = 0; i < N2; ++i)
            v[i] = callable(v[i]);
        return v;
    }

    constexpr friend inline floatN min(floatN v0, floatN v1) {
        return spread(v0, v1, LIFT(std::min));
    }

    constexpr friend inline floatN max(floatN v0, floatN v1) {
        return spread(v0, v1, LIFT(std::max));
    }

    inline floatN abs() const {
        return spread(LIFT(std::abs));
    }

    inline floatN sqrt() const {
        return spread(LIFT(std::sqrt));
    }

    constexpr inline floatN clamp(float min = 0.0f, float max = 1.0f) const {
        return spread([min, max](float x) { return std::clamp(x, min, max); });
    }

    template<typename... Args> requires (std::is_arithmetic_v<Args> && ...)
    constexpr inline floatN replace(Args... args) const {
        floatN v = *this;
        size_t i = 0;
        ((v.data[i++] = args), ...);
        return v;
    }

    template<size_t N2> requires (N2 <= N)
    constexpr inline floatN replace(const floatN<N2>& v2) const {
        floatN v = *this;
        for(size_t i = 0; i < std::min(N, N2); ++i)
            v[i] = v2[i];
        return v;
    }
};

using float2 = floatN<2>;
using float3 = floatN<3>;
using float4 = floatN<4>;

template<size_t N>
constexpr static inline float linear_rgb_to_luminance(floatN<N> linearRGB) requires (N >= 3) {
    return 0.2126f * linearRGB[0] + 0.7152f * linearRGB[1] + 0.0722f * linearRGB[2];
}

template<size_t N>
static inline floatN<N> srgb_to_linear_rgb(floatN<N> sRGB) requires (N >= 3) {
    return sRGB.spread(LIFT(FLIP::srgb_to_linear_rgb));
}

template<size_t N>
static inline floatN<N> linear_rgb_to_srgb(floatN<N> RGB) requires (N >= 3) {
    return RGB.spread(LIFT(FLIP::linear_rgb_to_srgb));
}

template<size_t N>
constexpr static inline floatN<N> linear_rgb_to_xyz(const floatN<N>& RGB) requires (N >= 3) {
    // Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
    // Assumes D65 standard illuminant.
    constexpr float a11 = 10135552.0f / 24577794.0f;
    constexpr float a12 = 8788810.0f / 24577794.0f;
    constexpr float a13 = 4435075.0f / 24577794.0f;
    constexpr float a21 = 2613072.0f / 12288897.0f;
    constexpr float a22 = 8788810.0f / 12288897.0f;
    constexpr float a23 = 887015.0f / 12288897.0f;
    constexpr float a31 = 1425312.0f / 73733382.0f;
    constexpr float a32 = 8788810.0f / 73733382.0f;
    constexpr float a33 = 70074185.0f / 73733382.0f;

    const float X = a11 * RGB.r() + a12 * RGB.g() + a13 * RGB.b();
    const float Y = a21 * RGB.r() + a22 * RGB.g() + a23 * RGB.b();
    const float Z = a31 * RGB.r() + a32 * RGB.g() + a33 * RGB.b();
    if constexpr(N >= 4) {
        // Pretend alpha is like luminance (Y) of a gray value and store in A
        const float A = a21 * RGB.a() + a22 * RGB.a() + a23 * RGB.a();
        return RGB.replace(X, Y, Z, A);
    }
    return RGB.replace(X, Y, Z);
}

template<size_t N>
constexpr static inline floatN<N> xyz_to_linear_rgb(const floatN<N>& XYZ) requires (N >= 3) {
    // Return values in linear RGB, assuming D65 standard illuminant.
    constexpr float a11 = 3.241003275f;
    constexpr float a12 = -1.537398934f;
    constexpr float a13 = -0.498615861f;
    constexpr float a21 = -0.969224334f;
    constexpr float a22 = 1.875930071f;
    constexpr float a23 = 0.041554224f;
    constexpr float a31 = 0.055639423f;
    constexpr float a32 = -0.204011202f;
    constexpr float a33 = 1.057148933f;

    float R = a11 * XYZ.x() + a12 * XYZ.y() + a13 * XYZ.z();
    float G = a21 * XYZ.x() + a22 * XYZ.y() + a23 * XYZ.z();
    float B = a31 * XYZ.x() + a32 * XYZ.y() + a33 * XYZ.z();
    if constexpr(N >= 4) {
        // Pretend alpha is purely luminance (Y), so recover it from the average luminance RGB
        const float A = (a12 * XYZ.a() + a22 * XYZ.a() + a32 * XYZ.a()) / 3.f;
        return XYZ.replace(R, G, B, A);
    }
    return XYZ.replace(R, G, B);
}

template<size_t N>
static inline floatN<N> xyz_to_cielab(const floatN<N>& XYZ, const floatN<3>& invReferenceIlluminant = INV_DEFAULT_ILLUMINANT) requires (N >= 3) {
    constexpr float delta = 6.0f / 29.0f;
    constexpr float deltaSquare = delta * delta;
    constexpr float deltaCube = delta * deltaSquare;
    constexpr float factor = 1.0f / (3.0f * deltaSquare);
    constexpr float term = 4.0f / 29.0f;

    // The default illuminant is D65.
    floatN<3> XYZ2 = XYZ;
    XYZ2 *= invReferenceIlluminant;
    XYZ2.x() = (XYZ2.x() > deltaCube ? std::pow(XYZ2.x(), 1.0f / 3.0f) : factor * XYZ2.x() + term);
    XYZ2.y() = (XYZ2.y() > deltaCube ? std::pow(XYZ2.y(), 1.0f / 3.0f) : factor * XYZ2.y() + term);
    XYZ2.z() = (XYZ2.z() > deltaCube ? std::pow(XYZ2.z(), 1.0f / 3.0f) : factor * XYZ2.z() + term);
    const float L = 116.0f * XYZ2.y() - 16.0f;
    const float a = 500.0f * (XYZ2.x() - XYZ2.y());
    const float b = 200.0f * (XYZ2.y() - XYZ2.z());

    if constexpr(N >= 4) {
        const float A = 116.0f * XYZ.a() - 16.0f;
        return XYZ.replace(L, a, b, A);
    }
    return XYZ.replace(L, a, b);
}

template<size_t N>
constexpr static inline floatN<N> cielab_to_xyz(const floatN<N>& Lab, const floatN<3>& referenceIlluminant = DEFAULT_ILLUMINANT) requires (N >= 3) {
    // The default illuminant is D65.
    float Y = (Lab[0] + 16.0f) / 116.0f;
    float X = Lab[1] / 500.0f + Y;
    float Z = Y - Lab[2] / 200.0f;

    const float delta = 6.0f / 29.0f;
    const float factor = 3.0f * delta * delta;
    const float term = 4.0f / 29.0f;
    floatN<3> XYZ;
    XYZ[0] = (X > delta ? X * X * X : (X - term) * factor);
    XYZ[1] = (Y > delta ? Y * Y * Y : (Y - term) * factor);
    XYZ[2] = (Z > delta ? Z * Z * Z : (Z - term) * factor);
    XYZ *= referenceIlluminant;
    if constexpr(N >= 4) {
        const float A = (Lab[3] + 16.0f) / 116.0f;
        return Lab.replace(XYZ.x(), XYZ.y(), XYZ.z(), A);
    }
    return Lab.replace(XYZ);
}

template<size_t N>
constexpr static inline floatN<N> xyz_to_ycxcz(const floatN<N>& XYZ, const floatN<3>& invReferenceIlluminant = INV_DEFAULT_ILLUMINANT) requires (N >= 3) {
    // The default illuminant is D65.
    floatN<3> XYZ2 = XYZ;
    XYZ2 *= invReferenceIlluminant;
    const float Y = 116.0f * XYZ2[1] - 16.0f;
    const float Cx = 500.0f * (XYZ2[0] - XYZ2[1]);
    const float Cz = 200.0f * (XYZ2[1] - XYZ2[2]);
    if constexpr(N >= 4) {
        const float A = 116.0f * XYZ[3] - 16.0f;
        return XYZ.replace(Y, Cx, Cz, A);
    }
    return XYZ.replace(Y, Cx, Cz);
}

template<size_t N>
constexpr static inline floatN<N> ycxycz_to_xyz(const floatN<N>& YCxCz, const floatN<3>& referenceIlluminant = DEFAULT_ILLUMINANT) requires (N >= 3) {
    // The default illuminant is D65.
    const float Y = (YCxCz[0] + 16.0f) / 116.0f;
    const float Cx = YCxCz[1] / 500.0f;
    const float Cz = YCxCz[2] / 200.0f;
    float X = Y + Cx;
    float Z = Y - Cz;
    floatN<3> XYZ(X, Y, Z);
    XYZ *= referenceIlluminant;
    if constexpr(N >= 4) {
        const float A = (YCxCz[3] + 16.0f) / 116.0f;
        return YCxCz.replace(XYZ.x(), XYZ.y(), XYZ.z(), A);
    }
    return YCxCz.replace(XYZ);
}

template<size_t N>
constexpr static inline float ycxcz_to_grayscale(const floatN<N>& YCxCz) {
    return (YCxCz[0] + 16.0f) / 116.0f; // Make it [0,1].
}

}
