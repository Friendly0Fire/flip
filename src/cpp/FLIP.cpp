/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: BSD-3-Clause
 */

 // Visualizing and Communicating Errors in Rendered Images
 // Ray Tracing Gems II, 2021,
 // by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
 // Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

 // Visualizing Errors in Rendered High Dynamic Range Images
 // Eurographics 2021,
 // by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
 // Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

 // FLIP: A Difference Evaluator for Alternating Images
 // High Performance Graphics 2020,
 // by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
 // Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
 // Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

// Single header code initially by Pontus Ebelin (formerly Andersson) and Tomas Akenine-Moller.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

#include "util.h"
#include "vecs.h"
#include "gaussian.h"
#include "tensor.h"
#include "FLIP.h"

#include <image.h>

namespace FLIP {

struct parameters {
    float ppd = calculate_ppd(0.7f, 3840.0f, 0.7f); // Populate PPD with default values based on 0.7 meters = distance to screen, 3840 pixels screen width, 0.7 meters monitor width.
    exposure_range exposure;
    int num_exposures = -1;                                                   // Used when the input is HDR.
    tonemapper tonemapper = tonemapper::aces;                                 // Default tonemapper (used for HDR).
};

static constexpr struct FLIP_constants {
    float gqc = 0.7f;
    float gpc = 0.4f;
    float gpt = 0.95f;
    float gw = 0.082f;
    float gqf = 0.5f;
} FLIPConstants;

constexpr inline float hunt(const float luminance, const float chrominance) {
    return 0.01f * luminance * chrominance;
}

template<size_t N> requires (N >= 3)
constexpr inline floatN<N> hunt(const floatN<N>& val) {
    return val.replace_at(1, hunt(val[0], val[1]), hunt(val[0], val[2]));
}

template<size_t N> requires (N >= 3)
inline float hy_ab(const floatN<N>& refPixel, const floatN<N>& testPixel) {
    float cityBlockDistanceL = std::fabs(refPixel[0] - testPixel[0]);
    float euclideanDistanceAB = std::sqrt(square(refPixel[1] - testPixel[1]) + square(refPixel[2] - testPixel[2]));
    // If present, alpha is considered equivalent to a grayscale color:
    // In L*a*b* space, L* represents luminance and a* and b* are effectively zero for a grayscale value
    // Therefore, alpha distance is represented purely by L* distance
    float cityBlockDistanceA = N >= 4 ? std::fabs(refPixel[3] - testPixel[3]) : 0.f;
    return cityBlockDistanceL + cityBlockDistanceA + euclideanDistanceAB;
}

inline float max_distance(float gqc) {
    static const auto greenLab = xyz_to_cielab(linear_rgb_to_xyz(floatN<3>(0.0f, 1.0f, 0.0f)));
    static const auto blueLab = xyz_to_cielab(linear_rgb_to_xyz(floatN<3>(0.0f, 0.0f, 1.0f)));
    static const auto greenLabHunt = hunt(greenLab);
    static const auto blueLabHunt = hunt(blueLab);
    return std::pow(hy_ab(greenLabHunt, blueLabHunt), gqc);
}



template<typename T>
static void set_max_exposure(image<T>& input, image<float>& errorMap, image<float>& exposureMap, float exposure) {
#pragma omp parallel for
    for(int y = 0; y < input.get_height(); y++) {
        for(int x = 0; x < input.get_width(); x++) {
            const float srcValue = errorMap.get(x, y);
            const float dstValue = input.get(x, y);

            if(srcValue > dstValue) {
                exposureMap.set(x, y, exposure);
                input.set(x, y, srcValue);
            }
        }
    }
}

template<typename T>
static void expose(image<T>& input, float level) {
    const float m = std::pow(2.0f, level);
#pragma omp parallel for
    for(int y = 0; y < input.get_height(); y++) {
        for(int x = 0; x < input.get_width(); x++) {
            input.set(x, y, input.get(x, y) * m);
        }
    }
}

template<typename T>
static exposure_range get_exposure_range(const image<T>& input, tonemapper toneMapper) {
    const float* tc = ToneMappingCoefficients[std::to_underlying(toneMapper)];
    constexpr float t = 0.85f;
    const float a = tc[0] - t * tc[3];
    const float b = tc[1] - t * tc[4];
    const float c = tc[2] - t * tc[5];

    float xMin = 0.0f;
    float xMax = 0.0f;
    solve_second_degree(xMin, xMax, a, b, c);

    float Ymin = 1e30f;
    float Ymax = -1e30f;
    std::vector<float> luminances;
    luminances.reserve(input.dims_.x() * input.dims_.y());
    for(int y = 0; y < input.dims_.y(); y++) {
        for(int x = 0; x < input.dims_.x(); x++) {
            float luminance = linear_rgb_to_luminance(input.get(x, y));
            luminances.push_back(luminance);
            if(luminance != 0.0f) {
                Ymin = std::min(luminance, Ymin);
            }
            Ymax = std::max(luminance, Ymax);
        }
    }

    const auto medianLocation = static_cast<std::ptrdiff_t>(luminances.size() / 2);
    std::ranges::nth_element(luminances, luminances.begin() + medianLocation);
    float Ymedian = luminances[medianLocation];
    Ymedian = std::max(Ymedian, std::numeric_limits<float>::epsilon()); // Avoid median = 0 when more than half of the image's pixels are black.

    return { .min = log2(xMax / Ymax), .max = log2(xMax / Ymedian) };
}

// For details, see separatedConvolutions.pdf in the FLIP repository:
// https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
struct spatial_filters {
    image<float3> filterYCx;
    image<float3> filterCz;
    spatial_filters(int width)
        : filterYCx(width, 1), filterCz(width, 1) {}
};
static spatial_filters generate_spatial_filters(float ppd) {
    const float deltaX = 1.0f / ppd;
    const int filterRadius = calculateSpatialFilterRadius(ppd);
    const int filterWidth = 2 * filterRadius + 1;
    spatial_filters filters(filterWidth);

    float3 filterSumYCx = 0.f;
    float3 filterSumCz = 0.f;

    for(int x = 0; x < filterWidth; x++) {
        const float ix = (static_cast<float>(x) - static_cast<float>(filterRadius)) * deltaX;

        const float ix2 = ix * ix;
        const float gY = Gaussian(ix2, GaussianConstants.a1.x(), GaussianConstants.b1.x());
        const float gCx = Gaussian(ix2, GaussianConstants.a1.y(), GaussianConstants.b1.y());
        const float gCz1 = GaussianSqrt(ix2, GaussianConstants.a1.z(), GaussianConstants.b1.z());
        const float gCz2 = GaussianSqrt(ix2, GaussianConstants.a2.z(), GaussianConstants.b2.z());
        const float3 valueYCx(gY, gCx, 0.0f);
        const float3 valueCz(gCz1, gCz2, 0.0f);
        filters.filterYCx.set(x, 0, valueYCx);
        filters.filterCz.set(x, 0, valueCz);
        filterSumYCx += valueYCx;
        filterSumCz += valueCz;
    }

    // Normalize weights.
    const float3 normFactorYCx = { 1.0f / filterSumYCx.x(), 1.0f / filterSumYCx.y(), 1.0f };
    const float normFactorCz = 1.0f / std::sqrt(filterSumCz.x() * filterSumCz.x() + filterSumCz.y() * filterSumCz.y());
    for(int x = 0; x < filterWidth; x++) {
        const float3 pYCx = filters.filterYCx.get(x, 0);
        const float3 pCz = filters.filterCz.get(x, 0);

        filters.filterYCx.set(x, 0, float3(pYCx.x() * normFactorYCx.x(), pYCx.y() * normFactorYCx.y(), 0.0f));
        filters.filterCz.set(x, 0, float3(pCz.x() * normFactorCz, pCz.y() * normFactorCz, 0.0f));
    }

    return filters;
}

// For details, see separatedConvolutions.pdf in the FLIP repository:
// https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
image<float3> generate_feature_filter(const float ppd) {
    const float stdDev = 0.5f * FLIPConstants.gw * ppd;
    const int radius = static_cast<int>(std::ceil(3.0f * stdDev));
    const int width = 2 * radius + 1;

    image<float3> filter(width, 1);

    float gSum = 0.0f;
    float dgSumNegative = 0.0f;
    float dgSumPositive = 0.0f;
    float ddgSumNegative = 0.0f;
    float ddgSumPositive = 0.0f;

    for(int x = 0; x < width; x++) {
        const int xx = x - radius;

        const float g = Gaussian(static_cast<float>(xx), stdDev);
        gSum += g;

        // 1st derivative.
        const float dg = -static_cast<float>(xx) * g;
        if(dg > 0.0f)
            dgSumPositive += dg;
        else
            dgSumNegative -= dg;

        // 2nd derivative.
        const float ddg = (static_cast<float>(xx) * static_cast<float>(xx) / (stdDev * stdDev) - 1.0f) * g;
        if(ddg > 0.0f)
            ddgSumPositive += ddg;
        else
            ddgSumNegative -= ddg;

        filter.set(x, 0, float3(g, dg, ddg));
    }

    // Normalize weights (Gaussian weights should sum to 1; positive and negative weights of 1st and 2nd derivative should sum to 1 and -1, respectively).
    for(int x = 0; x < width; x++) {
        float3 p = filter.get(x, 0);

        filter.set(x, 0, float3(p.x() / gSum, p.y() / (p.y() > 0.0f ? dgSumPositive : dgSumNegative), p.z() / (p.z() > 0.0f ? ddgSumPositive : ddgSumNegative)));
    }

    return filter;
}

// Performs spatial filtering (and clamps the results) on both the reference and test image at the same time (for better performance).
// Filtering has been changed to separable filtering for better performance. For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
// https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
// After filtering, compute color differences. referenceImage and testImage are expected to be in YCxCz space.
template<typename T>
image<float> color_difference(const image<T>& referenceImage, const image<T>& testImage, const spatial_filters& filters) {
    // Color difference constants
    const float cmax = max_distance(FLIPConstants.gqc);
    const float pccmax = FLIPConstants.gpc * cmax;

    const int halfFilterWidth = filters.filterYCx.get_width() / 2; // YCx and Cz filters are the same size.

    image<T> intermediateYCxImageReference(referenceImage.get_dimensions());
    image<T> intermediateYCxImageTest(referenceImage.get_dimensions());
    image<T> intermediateCzImageReference(referenceImage.get_dimensions());
    image<T> intermediateCzImageTest(referenceImage.get_dimensions());
    image<float> difference(referenceImage.get_dimensions());

    const int w = referenceImage.get_width();
    const int h = referenceImage.get_height();

    // Filter in x direction.
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            T intermediateYCxReference = 0.f;
            T intermediateYCxTest = 0.f;
            T intermediateCzReference = 0.f;
            T intermediateCzTest = 0.f;

            for(int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++) {
                const int xx = std::min(std::max(0, x + ix), w - 1);

                const T& weightsYCx = filters.filterYCx.get(ix + halfFilterWidth, 0);
                const T& weightsCz = filters.filterCz.get(ix + halfFilterWidth, 0);
                const T& referenceColor = referenceImage.get(xx, y);
                const T& testColor = testImage.get(xx, y);

                for(int i = 0; i < 2; ++i) {
                    intermediateYCxReference[i] += weightsYCx[i] * referenceColor[i];
                    intermediateYCxTest[i] += weightsYCx[i] * testColor[i];
                    intermediateCzReference[i] += weightsCz[i] * referenceColor[2];
                    intermediateCzTest[i] += weightsCz[i] * testColor[2];
                }
                
                if constexpr(T::count == 4) {
                    intermediateYCxReference[3] += weightsYCx[0] * referenceColor[3];
                    intermediateYCxTest[3] += weightsYCx[0] * testColor[3];
                }
            }

            intermediateYCxImageReference.set(x, y, intermediateYCxReference);
            intermediateYCxImageTest.set(x, y, intermediateYCxTest);
            intermediateCzImageReference.set(x, y, intermediateCzReference);
            intermediateCzImageTest.set(x, y, intermediateCzTest);
        }
    }

    // Filter in y direction.
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            T filteredYCxReference = 0.f;
            T filteredYCxTest = 0.f;
            T filteredCzReference = 0.f;
            T filteredCzTest = 0.f;

            for(int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++) {
                const int yy = std::min(std::max(0, y + iy), h - 1);

                const float3& weightsYCx = filters.filterYCx.get(iy + halfFilterWidth, 0);
                const float3& weightsCz = filters.filterCz.get(iy + halfFilterWidth, 0);
                const T& intermediateYCxReference = intermediateYCxImageReference.get(x, yy);
                const T& intermediateYCxTest = intermediateYCxImageTest.get(x, yy);
                const T& intermediateCzReference = intermediateCzImageReference.get(x, yy);
                const T& intermediateCzTest = intermediateCzImageTest.get(x, yy);

                for(int i = 0; i < 2; ++i) {
                    filteredYCxReference[i] += weightsYCx[i] * intermediateYCxReference[i];
                    filteredYCxTest[i] += weightsYCx[i] * intermediateYCxTest[i];
                    filteredCzReference[i] += weightsCz[i] * intermediateCzReference[2];
                    filteredCzTest[i] += weightsCz[i] * intermediateCzTest[2];
                }

                if constexpr(T::count == 4) {
                    filteredYCxReference[3] += weightsYCx[0] * intermediateYCxReference[3];
                    filteredYCxTest[3] += weightsYCx[0] * intermediateYCxTest[3];
                }
            }

            // Clamp to [0,1] in linear RGB.
            T filteredYCxCzReference = filteredYCxReference;
            filteredYCxReference.z() = filteredCzReference.x() + filteredCzReference.y();
            T filteredYCxCzTest = filteredYCxTest;
            filteredYCxTest.z() = filteredCzTest.x() + filteredCzTest.y();
            filteredYCxCzReference = xyz_to_linear_rgb(ycxycz_to_xyz(filteredYCxCzReference)).clamp();
            filteredYCxCzTest = xyz_to_linear_rgb(ycxycz_to_xyz(filteredYCxCzTest)).clamp();

            // Move from linear RGB to CIELab.
            auto filteredCIELabReference = xyz_to_cielab(linear_rgb_to_xyz(filteredYCxCzReference));
            auto filteredCIELabTest = xyz_to_cielab(linear_rgb_to_xyz(filteredYCxCzTest));

            // Apply Hunt adjustment.
            filteredCIELabReference = hunt(filteredCIELabReference);
            filteredCIELabTest = hunt(filteredCIELabTest);

            float colorDifference = hy_ab(filteredCIELabReference, filteredCIELabTest);

            colorDifference = powf(colorDifference, FLIPConstants.gqc);

            // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
            // while the rest are mapped to the range (gpt, 1].
            if(colorDifference < pccmax) {
                colorDifference *= FLIPConstants.gpt / pccmax;
            } else {
                colorDifference = FLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - FLIPConstants.gpt);
            }
            difference.set(x, y, colorDifference);
        }
    }

    return difference;
}

// This includes convolution (using separable filtering) of grayRefImage and grayTestImage for both edge and point filtering.
// In addition, it computes the final FLIP error and stores in "this". referenceImage and testImage are expected to be in YCxCz space.
template<typename T>
void feature_difference_and_final_error(image<float>& resultImage, const image<T>& referenceImage, const image<T>& testImage, const image<float3>& featureFilter) {
    constexpr float normalizationFactor = 1.0f / std::numbers::sqrt2_v<float>;
    const int halfFilterWidth = featureFilter.get_width() / 2;      // The edge and point filters are of the same size.
    const int w = referenceImage.get_width();
    const int h = referenceImage.get_height();

    using intermediate_t = std::conditional_t<T::count == 4, floatN<6>, floatN<3>>;

    image<intermediate_t> intermediateFeaturesImageReference(referenceImage.get_dimensions());
    image<intermediate_t> intermediateFeaturesImageTest(referenceImage.get_dimensions());

    // Convolve in x direction (1st and 2nd derivative for filter in x direction, Gaussian in y direction).
    // For details, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
    // We filter both reference and test image simultaneously (for better performance).
    constexpr float oneOver116 = 1.0f / 116.0f;
    constexpr float sixteenOver116 = 16.0f / 116.0f;
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
            float xReference = 0.0f, xTest = 0.0f;

            float daReference = 0.0f, daTest = 0.0f, ddaReference = 0.0f, ddaTest = 0.0f;
            float aReference = 0.0f, aTest = 0.0f;

            for(int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++) {
                const int xx = std::min(std::max(0, x + ix), w - 1);

                const float3 featureWeights = featureFilter.get(ix + halfFilterWidth, 0);
                const T& yReference = referenceImage.get(xx, y);
                const T& yTest = testImage.get(xx, y);

                // Normalize the Y values to [0,1].
                const float yReferenceNormalized = yReference.x() * oneOver116 + sixteenOver116;
                const float yTestNormalized = yTest.x() * oneOver116 + sixteenOver116;

                // Image multiplied by Gaussian.
                xReference += featureWeights.x() * yReferenceNormalized;
                xTest += featureWeights.x() * yTestNormalized;

                // Image multiplied by 1st and 2nd x-derivatives of Gaussian.
                dxReference += featureWeights.y() * yReferenceNormalized;
                dxTest += featureWeights.y() * yTestNormalized;
                ddxReference += featureWeights.z() * yReferenceNormalized;
                ddxTest += featureWeights.z() * yTestNormalized;

                if constexpr(T::count == 4) {
                    const float aReferenceNormalized = yReference.w() * oneOver116 + sixteenOver116;
                    const float aTestNormalized = yTest.w() * oneOver116 + sixteenOver116;

                    // Image multiplied by Gaussian.
                    aReference += featureWeights.x() * aReferenceNormalized;
                    aTest += featureWeights.x() * aTestNormalized;

                    // Image multiplied by 1st and 2nd x-derivatives of Gaussian.
                    daReference += featureWeights.y() * aReferenceNormalized;
                    daTest += featureWeights.y() * aTestNormalized;
                    ddaReference += featureWeights.z() * aReferenceNormalized;
                    ddaTest += featureWeights.z() * aTestNormalized;
                }
            }

            if constexpr(T::count == 4) {
                intermediateFeaturesImageReference.set(x, y, intermediate_t(dxReference, ddxReference, xReference, daReference, ddaReference, aReference));
                intermediateFeaturesImageTest.set(x, y, intermediate_t(dxTest, ddxTest, xTest, daTest, ddaTest, aTest));
            } else {
                intermediateFeaturesImageReference.set(x, y, intermediate_t(dxReference, ddxReference, xReference));
                intermediateFeaturesImageTest.set(x, y, intermediate_t(dxTest, ddxTest, xTest));
            }
        }
    }

    // Convolve in y direction (1st and 2nd derivative for filter in y direction, Gaussian in x direction), then compute difference.
    // For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
    // We filter both reference and test image simultaneously (for better performance).
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            float dxReference = 0.0f, dxTest = 0.0f, ddxReference = 0.0f, ddxTest = 0.0f;
            float dyReference = 0.0f, dyTest = 0.0f, ddyReference = 0.0f, ddyTest = 0.0f;

            float daReference = 0.0f, daTest = 0.0f, ddaReference = 0.0f, ddaTest = 0.0f;
            float dbReference = 0.0f, dbTest = 0.0f, ddbReference = 0.0f, ddbTest = 0.0f;

            for(int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++) {
                const int yy = std::min(std::max(0, y + iy), h - 1);

                const float3& featureWeights = featureFilter.get(iy + halfFilterWidth, 0);
                const intermediate_t& intermediateFeaturesReference = intermediateFeaturesImageReference.get(x, yy);
                const intermediate_t& intermediateFeatureTest = intermediateFeaturesImageTest.get(x, yy);

                // Intermediate images (1st and 2nd derivative in x) multiplied by Gaussian.
                dxReference += featureWeights.x() * intermediateFeaturesReference.x();
                dxTest += featureWeights.x() * intermediateFeatureTest.x();
                ddxReference += featureWeights.x() * intermediateFeaturesReference.y();
                ddxTest += featureWeights.x() * intermediateFeatureTest.y();

                // Intermediate image (Gaussian) multiplied by 1st and 2nd y-derivatives of Gaussian.
                dyReference += featureWeights.y() * intermediateFeaturesReference.z();
                dyTest += featureWeights.y() * intermediateFeatureTest.z();
                ddyReference += featureWeights.z() * intermediateFeaturesReference.z();
                ddyTest += featureWeights.z() * intermediateFeatureTest.z();

                if constexpr(T::count == 4) {
                    // Intermediate images (1st and 2nd derivative in x) multiplied by Gaussian.
                    daReference += featureWeights.x() * intermediateFeaturesReference[3];
                    daTest += featureWeights.x() * intermediateFeatureTest[3];
                    ddaReference += featureWeights.x() * intermediateFeaturesReference[4];
                    ddaTest += featureWeights.x() * intermediateFeatureTest[4];

                    // Intermediate image (Gaussian) multiplied by 1st and 2nd y-derivatives of Gaussian.
                    dbReference += featureWeights.y() * intermediateFeaturesReference[5];
                    dbTest += featureWeights.y() * intermediateFeatureTest[5];
                    ddbReference += featureWeights.z() * intermediateFeaturesReference[5];
                    ddbTest += featureWeights.z() * intermediateFeatureTest[5];
                }
            }

            const float edgeValueRef = std::sqrt(dxReference * dxReference + dyReference * dyReference);
            const float edgeValueTest = std::sqrt(dxTest * dxTest + dyTest * dyTest);
            const float pointValueRef = std::sqrt(ddxReference * ddxReference + ddyReference * ddyReference);
            const float pointValueTest = std::sqrt(ddxTest * ddxTest + ddyTest * ddyTest);

            const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
            const float pointDifference = std::abs(pointValueRef - pointValueTest);


            const float edgeValueRefA = std::sqrt(daReference * daReference + dbReference * dbReference);
            const float edgeValueTestA = std::sqrt(daTest * daTest + dbTest * dbTest);
            const float pointValueRefA = std::sqrt(ddaReference * ddaReference + ddbReference * ddbReference);
            const float pointValueTestA = std::sqrt(ddaTest * ddaTest + ddbTest * ddbTest);

            const float edgeDifferenceA = std::abs(edgeValueRefA - edgeValueTestA);
            const float pointDifferenceA = std::abs(pointValueRefA - pointValueTestA);

            const float featureDifference = std::pow(normalizationFactor * std::max({ edgeDifference, pointDifference, edgeDifferenceA, pointDifferenceA }), FLIPConstants.gqf);
            const float colorDifference = resultImage.get(x, y);

            const float errorFLIP = std::pow(colorDifference, 1.0f - featureDifference);

            resultImage.set(x, y, errorFLIP);
        }
    }
}

template<typename T>
image<float> FLIP_ldr(image<T>& reference, image<T>& test, float ppd)     // Both reference and test are assumed to be in linear RGB.
{
    // Transform from linear RGB to YCxCz.
    reference.linear_rgb_to_ycxcz();
    test.linear_rgb_to_ycxcz();

    // Prepare separated spatial filters. Because the filter for the Blue-Yellow channel is a sum of two Gaussians, we need to separate the spatial filter into two
    // (YCx for the Achromatic and Red-Green channels and Cz for the Blue-Yellow channel).
    const auto spatialFilters = generate_spatial_filters(ppd);

    // Prepare separated feature (edge/point) detection filters.
    const auto featureFilter = generate_feature_filter(ppd);

    // The next call performs spatial filtering on both the reference and test image at the same time (for better performance).
    // It then computes the color difference between the images.
    auto difference = color_difference(reference, test, spatialFilters);

    // The following call convolves referenceImage and testImage with the edge and point detection filters and performs additional
    // computations for the final feature differences, and then computes the final FLIP error and stores in "this".
    feature_difference_and_final_error(difference, reference, test, featureFilter);

    return difference;
}

#include "maps.inl"

static const image<float3> MagmaMap{ MapMagma };
static const image<float3> ViridisMap{ MapViridis };

/** Main function for computing (the image metric called) FLIP between a reference image and a test image.
 *  See FLIP-tool.cpp for usage of this function.
 *
 * @param[in] referenceImageInput Reference input image. For LDR, the content should be in [0,1]. Input is expected in linear RGB.
 * @param[in] testImageInput Test input image. For LDR, the content should be in [0,1]. Input is expected in linear RGB.
 * @param[in] useHDR Set to true if the input images are to be considered containing HDR content, i.e., not necessarily in [0,1].
 * @param[in,out] parameters Contains parameters (e.g., PPD, exposure settings,etc). If the exposures have not been set by the user, then those will be computed (and returned).
 * @param[out] errorMapFLIPOutput The FLIP error image in [0,1], a single channel (grayscale).
 * @param[out] maxErrorExposureMapOutput Exposure map output (only for HDR content).
 * @param[in] returnIntermediateLDRFLIPImages True if the next argument should be filled in by evaluate().
 * @param[out] intermediateLDRFLIPImages A list of temporary output LDR-FLIP error maps (in grayscale) from HDR-FLIP.
               See explanation of the errorMapFLIPOutput parameter for how to convert the maps to magma.
 * @param[in] returnIntermediateLDRImages True if the next argument should be filled in by evaluate().
 * @param[out] intermediateLDRImages A list of temporary tonemapped output LDR images (in linear RGB) from HDR-FLIP. Images in this order: Ref0, Test0, Ref1, Test1,...
 */
template<typename T>
evaluate_status evaluate(const image<T>& referenceImageInput, const image<T>& testImageInput,
    const bool useHDR, parameters& parameters, image<float>& errorMapFLIPOutput, image<float>& maxErrorExposureMapOutput) {
    image<T> referenceImage(referenceImageInput.get_width(), referenceImageInput.get_height());
    image<T> testImage(referenceImageInput.get_width(), referenceImageInput.get_height());
    referenceImage.copy(referenceImageInput);               // Make a copy, since FLIP_ldr() destroys the input images.
    testImage.copy(testImageInput);

    if(useHDR)         // Set parameters for HDR-FLIP.
    {
        // If startExposure/stopExposure are inf, they have not been set by the user. If so, compute from referenceImage.
        // See our paper about HDR-FLIP about the details.
        if(parameters.exposure.unbounded()) {
            const auto exposureRange = referenceImage.get_exposure_range(parameters.tonemapper);
            if(std::isinf(parameters.exposure.min))
                parameters.exposure.min = exposureRange.min;
            if(std::isinf(parameters.exposure.max))
                parameters.exposure.max = exposureRange.max;
        }
        if(parameters.exposure.min > parameters.exposure.max) {
            return evaluate_status::invalid_exposure_range;
        }
        if(parameters.num_exposures == -1)  // -1 means it has not been set by the user, so then we compute it.
            parameters.num_exposures = static_cast<int>(std::max(2.0f, std::ceil(parameters.exposure.range())));
    }

    if(useHDR) {   // Compute HDR-FLIP.
        image<T> rImage(referenceImage.get_dimensions());
        image<T> tImage(referenceImage.get_dimensions());

        float exposureStepSize = parameters.exposure.range() / static_cast<float>(parameters.num_exposures - 1);
        for(int i = 0; i < parameters.num_exposures; i++) {
            const float exposure = parameters.exposure.min + static_cast<float>(i) * exposureStepSize;
            rImage.copy(referenceImage);
            tImage.copy(testImage);
            rImage.expose(exposure);
            tImage.expose(exposure);
            rImage.apply_tonemap(parameters.tonemapper);
            tImage.apply_tonemap(parameters.tonemapper);
            rImage.clamp();
            tImage.clamp();
            auto tmpErrorMap = FLIP_ldr(rImage, tImage, parameters.ppd);
            set_max_exposure(errorMapFLIPOutput, tmpErrorMap, maxErrorExposureMapOutput, static_cast<float>(i) / static_cast<float>(parameters.num_exposures - 1));
        }
    } else {   // Compute LDR-FLIP.
        referenceImage.clamp();     // The input images should always be in [0,1], but we clamp them here to avoid any problems.
        testImage.clamp();
        errorMapFLIPOutput = FLIP_ldr(referenceImage, testImage, parameters.ppd);
    }

    return evaluate_status::success;
}

float get_mean_error(const image<float>& errorMapFLIPOutputImage) {
    float sum = 0.0f;
#pragma omp parallel for
    for(int y = 0; y < errorMapFLIPOutputImage.get_height(); y++) {
        for(int x = 0; x < errorMapFLIPOutputImage.get_width(); x++) {
            sum += errorMapFLIPOutputImage.get(x, y);
        }
    }
    return sum / static_cast<float>(errorMapFLIPOutputImage.get_width() * errorMapFLIPOutputImage.get_height());
}

image<float3> apply_color_map(const image<float>& errorMapFLIPOutputImage, color_map cm) {
    image<float3> magmaMappedFLIPImage(errorMapFLIPOutputImage.get_width(), errorMapFLIPOutputImage.get_height());
    magmaMappedFLIPImage.apply_color_map(errorMapFLIPOutputImage, cm == color_map::magma ? MagmaMap : ViridisMap);
    return magmaMappedFLIPImage;
}

}
