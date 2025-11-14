#pragma once
#include <image.h>
#include <vecs.h>

namespace FLIP {

struct parameters {
    float ppd = calculate_ppd(0.7f, 3840.0f, 0.7f); // Populate PPD with default values based on 0.7 meters = distance to screen, 3840 pixels screen width, 0.7 meters monitor width.
    exposure_range exposure;
    int num_exposures = -1;                                                   // Used when the input is HDR.
    tonemapper tonemapper = tonemapper::aces;                                 // Default tonemapper (used for HDR).
};

enum class color_map {
    magma,
    viridis
};

enum class evaluate_status {
    success,
    invalid_exposure_range,
};

/** Main function for computing (the image metric called) FLIP between a reference image and a test image.
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
    const bool useHDR, parameters& parameters, image<float>& errorMapFLIPOutput, image<float>& maxErrorExposureMapOutput);

float get_mean_error(const image<float>& errorMapFLIPOutputImage);
image<float3> apply_color_map(const image<float>& errorMapFLIPOutputImage, color_map cm);

}
