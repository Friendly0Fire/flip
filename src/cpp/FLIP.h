#pragma once
#include <image.h>
#include <vecs.h>

namespace FLIP {

enum class color_map {
    magma,
    viridis
};

enum class evaluate_status {
    success,
    invalid_exposure_range,
};



image<float3> apply_color_map(const image<float>& errorMapFLIPOutputImage, color_map cm);

}
