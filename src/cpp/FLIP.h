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

// Single header code by Pontus Ebelin (formerly Andersson) and Tomas Akenine-Moller.
//
// We provide the following evaluate() functions with different in/out parameters (see bottom of this file for more explanations):
//
// 1. evaluate(const bool useHDR, Parameters& parameters, image<float3>& referenceImageInput, image<float3>& testImageInput,
//                  image<float>& errorMapFLIPOutput, image<float>& maxErrorExposureMapOutput,
//                  const bool returnIntermediateLDRFLIPImages, std::vector<image<float>*>& intermediateLDRFLIPImages,
//                  const bool returnIntermediateLDRImages, std::vector<image<float3>*>& intermediateLDRImages)
//
//    # This is the one with most parameters and is used by FLIP-tool.cpp in main().
//    # See the function at the bottom of this file for detailed description of the parameters.
//
// 2. evaluate(const bool useHDR, Parameters& parameters, image<float3>& referenceImageInput, image<float3>& testImageInput,
//                  image<float>& errorMapFLIPOutput, image<float>& maxErrorExposureMap);
//
//    # We do not expect that many user will want the LDR-FLIP images and the tonemappe LDR images computed during HDR-FLIP, so provide this simpler function.
//
// 3.evaluate(const bool useHDR, Parameters& parameters, image<float3>& referenceImageInput, image<float3>& testImageInput,
//                 image<float>& errorMapFLIPOutput);
//
//    # This one also excludes the exposure map for HDR-FLIP, in case it is not used.
//
// 4. evaluate(const bool useHDR, Parameters& parameters, const int imageWidth, const int imageHeight,
//                  const float* referenceThreeChannelImage, const float* testThreeChannelImage, const bool applyMagmaMapToOutput, float** errorMapFLIPOutput)
//
//    # An even simpler function that does not use any of our image classes to input the images.

#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <limits>

#include "util.h"
#include "maps.h"
#include "vecs.h"
#include "gaussian.h"
#include "tensor.h"

namespace FLIP {

struct parameters {
    parameters() = default;
    float ppd = calculate_ppd(0.7f, 3840.0f, 0.7f);            // Populate PPD with default values based on 0.7 meters = distance to screen, 3840 pixels screen width, 0.7 meters monitor width.
    exposure_range exposure;
    int num_exposures = -1;                                          // Used when the input is HDR.
    tonemapper tonemapper = tonemapper::aces;                                // Default tonemapper (used for HDR).
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

template<size_t N>
inline float hy_ab(const floatN<N>& refPixel, const floatN<N>& testPixel) {
    float cityBlockDistanceL = std::fabs(refPixel[0] - testPixel[0]);
    float euclideanDistanceAB = std::sqrt(square(refPixel[1] - testPixel[1]) + square(refPixel[2] - testPixel[2]));
    return cityBlockDistanceL + euclideanDistanceAB;
}

inline float max_distance(float gqc) {
    static const floatN greenLab = xyz_to_cielab(linear_rgb_to_xyz(floatN<3>(0.0f, 1.0f, 0.0f)));
    static const floatN blueLab = xyz_to_cielab(linear_rgb_to_xyz(floatN<3>(0.0f, 0.0f, 1.0f)));
    static const floatN greenLabHunt = floatN<3>(greenLab[0], hunt(greenLab[0], greenLab[1]), hunt(greenLab[0], greenLab[2]));
    static const floatN blueLabHunt = floatN<3>(blueLab[0], hunt(blueLab[0], blueLab[1]), hunt(blueLab[0], blueLab[2]));
    return std::pow(hy_ab(greenLabHunt, blueLabHunt), gqc);
}

template<typename T>
class image : public tensor<T> {
public:
    image() = default;

    image(const int width, const int height)
        : tensor<T>(width, height, 1) { }

    image(const int width, const int height, const T clearColor)
        : tensor<T>(width, height, 1, clearColor) { }

    image(const int3 dim, const T clearColor)
        : tensor<T>(dim.x(), dim.y(), 1) { }

    image(const int3 dim)
        : tensor<T>(dim.x(), dim.y(), 1) { }

    image(const image& image) {
        this->init(image.dims_);
        this->copy(image);
    }

    image(const tensor<T>& tensor, const int offset) {
        this->init(tensor.get_dimensions());
        this->copy(tensor, offset);
    }

    image(std::span<const T> input)
        : tensor<T>(input) { }

    image(std::span<const T> input, int width)
        : tensor<T>(input, width) { }

    image(std::span<const float> input) requires (!std::same_as<T, float>)
        : tensor<T>(input) { }

    image(std::span<const float> input, int width) requires (!std::same_as<T, float>)
        : tensor<T>(input, width) { }

    T get(int x, int y) const {
        return this->data_[this->index(x, y)];
    }

    void set(int x, int y, const T& value) {
        this->data_[this->index(x, y)] = value;
    }

    // For details, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf.
    static void set_spatial_filters(image<float3>& filterYCx, image<float3>& filterCz, float ppd, int filterRadius) {
        float deltaX = 1.0f / ppd;
        float3 filterSumYCx = { 0.0f, 0.0f, 0.0f };
        float3 filterSumCz = { 0.0f, 0.0f, 0.0f };
        int filterWidth = 2 * filterRadius + 1;

        for(int x = 0; x < filterWidth; x++) {
            const float ix = (static_cast<float>(x) - filterRadius) * deltaX;

            const float ix2 = ix * ix;
            const float gY = Gaussian(ix2, GaussianConstants.a1.x(), GaussianConstants.b1.x());
            const float gCx = Gaussian(ix2, GaussianConstants.a1.y(), GaussianConstants.b1.y());
            const float gCz1 = GaussianSqrt(ix2, GaussianConstants.a1.z(), GaussianConstants.b1.z());
            const float gCz2 = GaussianSqrt(ix2, GaussianConstants.a2.z(), GaussianConstants.b2.z());
            const float3 valueYCx(gY, gCx, 0.0f);
            const float3 valueCz(gCz1, gCz2, 0.0f);
            filterYCx.set(x, 0, valueYCx);
            filterCz.set(x, 0, valueCz);
            filterSumYCx += valueYCx;
            filterSumCz += valueCz;
        }

        // Normalize weights.
        const float3 normFactorYCx = { 1.0f / filterSumYCx.x(), 1.0f / filterSumYCx.y(), 1.0f };
        const float normFactorCz = 1.0f / std::sqrt(filterSumCz.x() * filterSumCz.x() + filterSumCz.y() * filterSumCz.y());
        for(int x = 0; x < filterWidth; x++) {
            const float3 pYCx = filterYCx.get(x, 0);
            const float3 pCz = filterCz.get(x, 0);

            filterYCx.set(x, 0, float3(pYCx.x() * normFactorYCx.x(), pYCx.y() * normFactorYCx.y(), 0.0f));
            filterCz.set(x, 0, float3(pCz.x() * normFactorCz, pCz.y() * normFactorCz, 0.0f));
        }
    }

    // For details, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
    static void set_feature_filter(image<float3>& filter, const float ppd) {
        const float stdDev = 0.5f * FLIPConstants.gw * ppd;
        const int radius = static_cast<int>(std::ceil(3.0f * stdDev));
        const int width = 2 * radius + 1;

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
    }

    // Performs spatial filtering (and clamps the results) on both the reference and test image at the same time (for better performance).
    // Filtering has been changed to separable filtering for better performance. For details on the convolution, see separatedConvolutions.pdf in the FLIP repository:
    // https://github.com/NVlabs/flip/blob/main/misc/separatedConvolutions.pdf
    // After filtering, compute color differences. referenceImage and testImage are expected to be in YCxCz space.
    void color_difference(const image<float3>& referenceImage, const image<float3>& testImage, const image<float3>& filterYCx, const image<float3>& filterCz) {
        // Color difference constants
        const float cmax = max_distance(FLIPConstants.gqc);
        const float pccmax = FLIPConstants.gpc * cmax;

        const int halfFilterWidth = filterYCx.get_width() / 2; // YCx and Cz filters are the same size.

        const int w = referenceImage.get_width();
        const int h = referenceImage.get_height();

        image<float3> intermediateYCxImageReference(w, h);
        image<float3> intermediateYCxImageTest(w, h);
        image<float3> intermediateCzImageReference(w, h);
        image<float3> intermediateCzImageTest(w, h);

        // Filter in x direction.
    #pragma omp parallel for
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                float3 intermediateYCxReference = { 0.0f, 0.0f, 0.0f };
                float3 intermediateYCxTest = { 0.0f, 0.0f, 0.0f };
                float3 intermediateCzReference = { 0.0f, 0.0f, 0.0f };
                float3 intermediateCzTest = { 0.0f, 0.0f, 0.0f };

                for(int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++) {
                    const int xx = std::min(std::max(0, x + ix), w - 1);

                    const float3 weightsYCx = filterYCx.get(ix + halfFilterWidth, 0);
                    const float3 weightsCz = filterCz.get(ix + halfFilterWidth, 0);
                    const float3 referenceColor = referenceImage.get(xx, y);
                    const float3 testColor = testImage.get(xx, y);

                    intermediateYCxReference += float3(weightsYCx.x() * referenceColor.x(), weightsYCx.y() * referenceColor.y(), 0.0f);
                    intermediateYCxTest += float3(weightsYCx.x() * testColor.x(), weightsYCx.y() * testColor.y(), 0.0f);
                    intermediateCzReference += float3(weightsCz.x() * referenceColor.z(), weightsCz.y() * referenceColor.z(), 0.0f);
                    intermediateCzTest += float3(weightsCz.x() * testColor.z(), weightsCz.y() * testColor.z(), 0.0f);
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
                float3 filteredYCxReference = { 0.0f, 0.0f, 0.0f };
                float3 filteredYCxTest = { 0.0f, 0.0f, 0.0f };
                float3 filteredCzReference = { 0.0f, 0.0f, 0.0f };
                float3 filteredCzTest = { 0.0f, 0.0f, 0.0f };

                for(int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++) {
                    const int yy = std::min(std::max(0, y + iy), h - 1);

                    const float3 weightsYCx = filterYCx.get(iy + halfFilterWidth, 0);
                    const float3 weightsCz = filterCz.get(iy + halfFilterWidth, 0);
                    const float3 intermediateYCxReference = intermediateYCxImageReference.get(x, yy);
                    const float3 intermediateYCxTest = intermediateYCxImageTest.get(x, yy);
                    const float3 intermediateCzReference = intermediateCzImageReference.get(x, yy);
                    const float3 intermediateCzTest = intermediateCzImageTest.get(x, yy);

                    filteredYCxReference += float3(weightsYCx.x() * intermediateYCxReference.x(), weightsYCx.y() * intermediateYCxReference.y(), 0.0f);
                    filteredYCxTest += float3(weightsYCx.x() * intermediateYCxTest.x(), weightsYCx.y() * intermediateYCxTest.y(), 0.0f);
                    filteredCzReference += float3(weightsCz.x() * intermediateCzReference.x(), weightsCz.y() * intermediateCzReference.y(), 0.0f);
                    filteredCzTest += float3(weightsCz.x() * intermediateCzTest.x(), weightsCz.y() * intermediateCzTest.y(), 0.0f);
                }

                // Clamp to [0,1] in linear RGB.
                float3 filteredYCxCzReference = float3(filteredYCxReference.x(), filteredYCxReference.y(), filteredCzReference.x() + filteredCzReference.y());
                float3 filteredYCxCzTest = float3(filteredYCxTest.x(), filteredYCxTest.y(), filteredCzTest.x() + filteredCzTest.y());
                filteredYCxCzReference = xyz_to_linear_rgb(ycxycz_to_xyz(filteredYCxCzReference)).clamp();
                filteredYCxCzTest = xyz_to_linear_rgb(ycxycz_to_xyz(filteredYCxCzTest)).clamp();

                // Move from linear RGB to CIELab.
                filteredYCxCzReference = xyz_to_cielab(linear_rgb_to_xyz(filteredYCxCzReference));
                filteredYCxCzTest = xyz_to_cielab(linear_rgb_to_xyz(filteredYCxCzTest));

                // Apply Hunt adjustment.
                filteredYCxCzReference.y() = hunt(filteredYCxCzReference.x(), filteredYCxCzReference.y());
                filteredYCxCzReference.z() = hunt(filteredYCxCzReference.x(), filteredYCxCzReference.z());
                filteredYCxCzTest.y() = hunt(filteredYCxCzTest.x(), filteredYCxCzTest.y());
                filteredYCxCzTest.z() = hunt(filteredYCxCzTest.x(), filteredYCxCzTest.z());

                float colorDifference = hy_ab(filteredYCxCzReference, filteredYCxCzTest);

                colorDifference = powf(colorDifference, FLIPConstants.gqc);

                // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
                // while the rest are mapped to the range (gpt, 1].
                if(colorDifference < pccmax) {
                    colorDifference *= FLIPConstants.gpt / pccmax;
                } else {
                    colorDifference = FLIPConstants.gpt + ((colorDifference - pccmax) / (cmax - pccmax)) * (1.0f - FLIPConstants.gpt);
                }
                this->set(x, y, colorDifference);
            }
        }
    }

    // This includes convolution (using separable filtering) of grayRefImage and grayTestImage for both edge and point filtering.
    // In addition, it computes the final FLIP error and stores in "this". referenceImage and testImage are expected to be in YCxCz space.
    void feature_difference_and_final_error(const image<float3>& referenceImage, const image<float3>& testImage, const image<float3>& featureFilter) {
        constexpr float normalizationFactor = 1.0f / std::numbers::sqrt2_v<float>;
        const int halfFilterWidth = featureFilter.get_width() / 2;      // The edge and point filters are of the same size.
        const int w = referenceImage.get_width();
        const int h = referenceImage.get_height();

        image<float3> intermediateFeaturesImageReference(w, h);
        image<float3> intermediateFeaturesImageTest(w, h);

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
                float gaussianFilteredReference = 0.0f, gaussianFilteredTest = 0.0f;

                for(int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++) {
                    const int xx = std::min(std::max(0, x + ix), w - 1);

                    const float3 featureWeights = featureFilter.get(ix + halfFilterWidth, 0);
                    const float yReference = referenceImage.get(xx, y).x();
                    const float yTest = testImage.get(xx, y).x();

                    // Normalize the Y values to [0,1].
                    const float yReferenceNormalized = yReference * oneOver116 + sixteenOver116;
                    const float yTestNormalized = yTest * oneOver116 + sixteenOver116;

                    // Image multiplied by 1st and 2nd x-derivatives of Gaussian.
                    dxReference += featureWeights.y() * yReferenceNormalized;
                    dxTest += featureWeights.y() * yTestNormalized;
                    ddxReference += featureWeights.z() * yReferenceNormalized;
                    ddxTest += featureWeights.z() * yTestNormalized;

                    // Image multiplied by Gaussian.
                    gaussianFilteredReference += featureWeights.x() * yReferenceNormalized;
                    gaussianFilteredTest += featureWeights.x() * yTestNormalized;
                }
                intermediateFeaturesImageReference.set(x, y, float3(dxReference, ddxReference, gaussianFilteredReference));
                intermediateFeaturesImageTest.set(x, y, float3(dxTest, ddxTest, gaussianFilteredTest));
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

                for(int iy = -halfFilterWidth; iy <= halfFilterWidth; iy++) {
                    const int yy = std::min(std::max(0, y + iy), h - 1);

                    const float3 featureWeights = featureFilter.get(iy + halfFilterWidth, 0);
                    const float3 intermediateFeaturesReference = intermediateFeaturesImageReference.get(x, yy);
                    const float3 intermediateFeatureTest = intermediateFeaturesImageTest.get(x, yy);

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
                }

                const float edgeValueRef = std::sqrt(dxReference * dxReference + dyReference * dyReference);
                const float edgeValueTest = std::sqrt(dxTest * dxTest + dyTest * dyTest);
                const float pointValueRef = std::sqrt(ddxReference * ddxReference + ddyReference * ddyReference);
                const float pointValueTest = std::sqrt(ddxTest * ddxTest + ddyTest * ddyTest);

                const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
                const float pointDifference = std::abs(pointValueRef - pointValueTest);

                const float featureDifference = std::pow(normalizationFactor * std::max(edgeDifference, pointDifference), FLIPConstants.gqf);
                const float colorDifference = this->get(x, y);

                const float errorFLIP = std::pow(colorDifference, 1.0f - featureDifference);

                this->set(x, y, errorFLIP);
            }
        }
    }

    void set_max_exposure(image<float>& errorMap, image<float>& exposureMap, float exposure) {
    #pragma omp parallel for
        for(int y = 0; y < this->get_height(); y++) {
            for(int x = 0; x < this->get_width(); x++) {
                const float srcValue = errorMap.get(x, y);
                const float dstValue = this->get(x, y);

                if(srcValue > dstValue) {
                    exposureMap.set(x, y, exposure);
                    this->set(x, y, srcValue);
                }
            }
        }
    }

    void expose(float level) {
        const float m = std::pow(2.0f, level);
    #pragma omp parallel for
        for(int y = 0; y < this->get_height(); y++) {
            for(int x = 0; x < this->get_width(); x++) {
                this->set(x, y, this->get(x, y) * m);
            }
        }
    }

    exposure_range get_exposure_range(tonemapper toneMapper) {
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
        luminances.reserve(this->dims_.x() * this->dims_.y());
        for(int y = 0; y < this->dims_.y(); y++) {
            for(int x = 0; x < this->dims_.x(); x++) {
                float luminance = linear_rgb_to_luminance(this->get(x, y));
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

    void FLIP_ldr(image<float3>& reference, image<float3>& test, float ppd)     // Both reference and test are assumed to be in linear RGB.
    {
        // Transform from linear RGB to YCxCz.
        reference.linear_rgb_to_ycxcz();
        test.linear_rgb_to_ycxcz();

        // Prepare separated spatial filters. Because the filter for the Blue-Yellow channel is a sum of two Gaussians, we need to separate the spatial filter into two
        // (YCx for the Achromatic and Red-Green channels and Cz for the Blue-Yellow channel).
        int spatialFilterRadius = calculateSpatialFilterRadius(ppd);
        int spatialFilterWidth = 2 * spatialFilterRadius + 1;
        image<float3> spatialFilterYCx(spatialFilterWidth, 1);
        image<float3> spatialFilterCz(spatialFilterWidth, 1);
        set_spatial_filters(spatialFilterYCx, spatialFilterCz, ppd, spatialFilterRadius);

        // The next call performs spatial filtering on both the reference and test image at the same time (for better performance).
        // It then computes the color difference between the images. "this" is an image<float> here, so we store the color difference in that image.
        this->color_difference(reference, test, spatialFilterYCx, spatialFilterCz);

        // Prepare separated feature (edge/point) detection filters.
        const float stdDev = 0.5f * FLIPConstants.gw * ppd;
        const int featureFilterRadius = int(std::ceil(3.0f * stdDev));
        int featureFilterWidth = 2 * featureFilterRadius + 1;
        image<float3> featureFilter(featureFilterWidth, 1);
        set_feature_filter(featureFilter, ppd);

        // The following call convolves referenceImage and testImage with the edge and point detection filters and performs additional
        // computations for the final feature differences, and then computes the final FLIP error and stores in "this".
        this->feature_difference_and_final_error(reference, test, featureFilter);
    }
};

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
               The user should map it using MapMagma if that is desired (with: errorMapWithMagma.colorMap(errorMapFLIP, magmaMap);)
 * @param[out] maxErrorExposureMapOutput Exposure map output (only for HDR content).
 * @param[in] returnIntermediateLDRFLIPImages True if the next argument should be filled in by evaluate().
 * @param[out] intermediateLDRFLIPImages A list of temporary output LDR-FLIP error maps (in grayscale) from HDR-FLIP.
               See explanation of the errorMapFLIPOutput parameter for how to convert the maps to magma.
 * @param[in] returnIntermediateLDRImages True if the next argument should be filled in by evaluate().
 * @param[out] intermediateLDRImages A list of temporary tonemapped output LDR images (in linear RGB) from HDR-FLIP. Images in this order: Ref0, Test0, Ref1, Test1,...
 */
static void evaluate(image<float3>& referenceImageInput, image<float3>& testImageInput,
    const bool useHDR, parameters& parameters, image<float>& errorMapFLIPOutput, image<float>& maxErrorExposureMapOutput,
    const bool returnIntermediateLDRFLIPImages, std::vector<image<float>*>& intermediateLDRFLIPImages,
    const bool returnIntermediateLDRImages, std::vector<image<float3>*>& intermediateLDRImages) {
    image<float3> referenceImage(referenceImageInput.get_width(), referenceImageInput.get_height());
    image<float3> testImage(referenceImageInput.get_width(), referenceImageInput.get_height());
    referenceImage.copy(referenceImageInput);               // Make a copy, since image::LDR_FLIP() destroys the input images.
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
            std::cout << "Start exposure must be smaller than stop exposure!\n";
            std::exit(-1);
        }
        if(parameters.num_exposures == -1)  // -1 means it has not been set by the user, so then we compute it.
            parameters.num_exposures = static_cast<int>(std::max(2.0f, std::ceil(parameters.exposure.range())));
    }

    if(useHDR)     // Compute HDR-FLIP.
    {
        image<float3> rImage(referenceImage.get_width(), referenceImage.get_height());
        image<float3> tImage(referenceImage.get_width(), referenceImage.get_height());
        image<float> tmpErrorMap(referenceImage.get_width(), referenceImage.get_height(), 0.0f);

        float exposureStepSize = parameters.exposure.range() / (parameters.num_exposures - 1);
        for(int i = 0; i < parameters.num_exposures; i++) {
            float exposure = parameters.exposure.min + i * exposureStepSize;
            rImage.copy(referenceImage);
            tImage.copy(testImage);
            rImage.expose(exposure);
            tImage.expose(exposure);
            rImage.apply_tonemap(parameters.tonemapper);
            tImage.apply_tonemap(parameters.tonemapper);
            rImage.clamp();
            tImage.clamp();
            if(returnIntermediateLDRImages) {
                intermediateLDRImages.push_back(new image<float3>(rImage));
                intermediateLDRImages.push_back(new image<float3>(tImage));
            }
            tmpErrorMap.FLIP_ldr(rImage, tImage, parameters.ppd);
            if(returnIntermediateLDRFLIPImages) {
                intermediateLDRFLIPImages.push_back(new image<float>(tmpErrorMap));
            }
            errorMapFLIPOutput.set_max_exposure(tmpErrorMap, maxErrorExposureMapOutput, float(i) / (parameters.num_exposures - 1));
        }
    } else    // Compute LDR-FLIP.
    {
        referenceImage.clamp();     // The input images should always be in [0,1], but we clamp them here to avoid any problems.
        testImage.clamp();
        errorMapFLIPOutput.FLIP_ldr(referenceImage, testImage, parameters.ppd);
    }
}

// This variant does not return any LDR images computed by HDR-FLIP and thus avoids two parameters (since using those is a rare use case).
static void evaluate(image<float3>& referenceImageInput, image<float3>& testImageInput,
    const bool useHDR, parameters& parameters, image<float>& errorMapFLIPOutput, image<float>& maxErrorExposureMapOutput) {
    std::vector<image<float>*> intermediateLDRFLIPImages;
    std::vector<image<float3>*> intermediateLDRImages;
    evaluate(referenceImageInput, testImageInput, useHDR, parameters, errorMapFLIPOutput, maxErrorExposureMapOutput, false, intermediateLDRFLIPImages, false, intermediateLDRImages);
}

// This variant does not return the exposure map, which may also be used quite seldom.
static void evaluate(image<float3>& referenceImageInput, image<float3>& testImageInput, const bool useHDR, parameters& parameters,
    image<float>& errorMapFLIPOutput) {
    image<float> maxErrorExposureMapOutput(referenceImageInput.get_width(), referenceImageInput.get_height());
    evaluate(referenceImageInput, testImageInput, useHDR, parameters, errorMapFLIPOutput, maxErrorExposureMapOutput);
}

/** A simplified function for computing (the image metric called) FLIP between a reference image and a test image, without the input images being defined using image, etc.
 *
 * Note that the user is responsible for deallocating the output image in the varible errorMapFLIPOutput. See the desciption of errorMapFLIPOutput below.
 *
 * @param[in] referenceThreeChannelImage Reference input image. For LDR, the content should be in [0,1]. The image is expected to have 3 floats per pixel and they
 *            are interleaved, i.e., they come in the order: R0G0B0, R1G1B1, etc. Input is expected to be in linear RGB.
 * @param[in] testThreeChannelImage Test input image. For LDR, the content should be in [0,1]. The image is expected to have 3 floats per pixel and they are interleaved.
              Input is expected to be in linear RGB.
 * @param[in] imageWidth Width of the reference and test images.
 * @param[in] imageHeight Height of the reference and test images.
 * @param[in,out] parameters Contains parameters (e.g., PPD, exposure settings,etc). If the exposures have not been set by the user, then those will be computed (and returned).
 * @param[in] useHDR Set to true if the input images are to be considered containing HDR content, i.e., not necessarily in [0,1].
 * @param[in] applyMagmaMapToOutput A boolean indicating whether the output should have the MagmaMap applied to it before the image is returned.
 * @param[in] computeMeanFLIPError Set to true if the mean FLIP error should be computed. If false, mean error is set to -1.
 * @param[out] meanFLIPError Mean FLIP error in the test (testThreeChannelImage) compared to the reference (referenceThreeChannelImage).
 * @param[out] errorMapFLIPOutput The computed FLIP error image is returned in this variable. If applyMagmaMapToOutput is true, the function will allocate
 *             three channels (and store the magma-mapped FLIP images in sRGB), and
 *             if it is false, only one channel will be allocated (and the FLIP error is returned in that grayscale image).
 *             Note that the user is responsible for deallocating the errorMapFLIPOutput image.
 */
static void evaluate(const float* referenceThreeChannelImage, const float* testThreeChannelImage,
    const int imageWidth, const int imageHeight, const bool useHDR, parameters& parameters,
    const bool applyMagmaMapToOutput, const bool computeMeanFLIPError, float& meanFLIPError, float** errorMapFLIPOutput) {
    image<float3> referenceImage(std::span(referenceThreeChannelImage, imageWidth * imageHeight), imageWidth);
    image<float3> testImage(std::span(testThreeChannelImage, imageWidth * imageHeight), imageWidth);
    image<float> errorMapFLIPOutputImage(imageWidth, imageHeight, 0.0f);

    evaluate(referenceImage, testImage, useHDR, parameters, errorMapFLIPOutputImage);

    // Compute mean FLIP error, if desired.
    if(computeMeanFLIPError) {
        float sum = 0.0f;

    #pragma omp parallel for
        for(int y = 0; y < errorMapFLIPOutputImage.get_height(); y++) {
            for(int x = 0; x < errorMapFLIPOutputImage.get_width(); x++) {
                sum += errorMapFLIPOutputImage.get(x, y);
            }
        }
        meanFLIPError = sum / (errorMapFLIPOutputImage.get_width() * errorMapFLIPOutputImage.get_height());
    }

    if(applyMagmaMapToOutput) {
        *errorMapFLIPOutput = new float[imageWidth * imageHeight * 3];
        image<float3> magmaMappedFLIPImage(imageWidth, imageHeight);
        magmaMappedFLIPImage.apply_color_map(errorMapFLIPOutputImage, MagmaMap);
        memcpy(*errorMapFLIPOutput, magmaMappedFLIPImage.get_data(), size_t(imageWidth) * imageHeight * sizeof(float) * 3);

    } else    // No MagmaMap applied, which means that we will return the gray scale image.
    {
        *errorMapFLIPOutput = new float[imageWidth * imageHeight];
        memcpy(*errorMapFLIPOutput, errorMapFLIPOutputImage.get_data(), size_t(imageWidth) * imageHeight * sizeof(float));
    }
}
}
