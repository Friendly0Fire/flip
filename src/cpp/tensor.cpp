#pragma once

#include "tensor.h"

namespace FLIP {

template<typename T>
void tensor<T>::init(const int3& dims, bool should_clear, const T& clear_color) {
    dims_ = dims;
    area_ = dims.x() * dims.y();
    volume_ = dims.x() * dims.y() * dims.z();

    data_.resize(volume_);

    if(should_clear) {
        clear(clear_color);
    }
}

template<typename T>
void tensor<T>::srgb_to_ycxcz() requires floatN_at_least<T, 3> {
    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                set(x, y, z, FLIP::xyz_to_ycxcz(FLIP::linear_rgb_to_xyz(FLIP::srgb_to_linear_rgb(get(x, y, z)))));
            }
        }
    }
}

template<typename T>
void tensor<T>::srgb_to_linear_rgb() requires floatN_at_least<T, 3> {
    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                set(x, y, z, FLIP::srgb_to_linear_rgb(get(x, y, z)));
            }
        }
    }
}

template<typename T>
void tensor<T>::linear_rgb_to_ycxcz() requires floatN_at_least<T, 3> {
    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                set(x, y, z, FLIP::xyz_to_ycxcz(FLIP::linear_rgb_to_xyz(get(x, y, z))));
            }
        }
    }
}

template<typename T>
void tensor<T>::linear_rgb_to_srgb() requires floatN_at_least<T, 3> {
    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                set(x, y, z, FLIP::linear_rgb_to_srgb(get(x, y, z)));
            }
        }
    }
}

template<typename T>
void tensor<T>::clamp(float low, float high) {
    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                if constexpr(std::same_as<T, float>)
                    set(x, y, z, std::clamp(get(x, y, z), low, high));
                else
                    set(x, y, z, get(x, y, z).clamp(low, high));
            }
        }
    }
}

template<typename T>
void tensor<T>::apply_tonemap(tonemapper tm) requires floatN_at_least<T, 3> {
    if(tm == tonemapper::reinhard) {
        for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    T color = get(x, y, z);
                    float luminance = linear_rgb_to_luminance(color);
                    float factor = 1.0f / (1.0f + luminance);
                    set(x, y, z, color * factor);
                }
            }
        }
        return;
    }

    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                const float* tc = ToneMappingCoefficients[std::to_underlying(tm)];
                float3 color = get(x, y, z);
                set(x, y, z, float3(((color * color) * tc[0] + color * tc[1] + tc[2]) / (color * color * tc[3] + color * tc[4] + tc[5])));
            }
        }
    }
}

template<typename T>
void tensor<T>::expand_grayscale(const tensor<float>& srcImage) {
    for(int z = 0; z < get_depth(); z++) {
#pragma omp parallel for
        for(int y = 0; y < get_height(); y++) {
            for(int x = 0; x < get_width(); x++) {
                set(x, y, z, T(srcImage.get(x, y, z)));
            }
        }
    }
}

template<typename T>
void tensor<T>::apply_color_map(const tensor<float>& srcImage, const tensor<float3>& colorMap) requires floatN_at_least<T, 3> {
    for(int z = 0; z < this->get_depth(); z++) {
    #pragma omp parallel for
        for(int y = 0; y < this->get_height(); y++) {
            for(int x = 0; x < this->get_width(); x++) {
                this->set(x, y, z, colorMap.get(static_cast<int>(srcImage.get(x, y, z) * 255.0f + 0.5f) % colorMap.get_width(), 0, 0));
            }
        }
    }
}

template class tensor<float>;
template class tensor<float2>;
template class tensor<float3>;
template class tensor<float4>;

}
