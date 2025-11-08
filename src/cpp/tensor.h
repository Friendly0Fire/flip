#pragma once
#include "vecs.h"

namespace FLIP {

template<typename T>
class tensor {
protected:
    int3 dims_ = { 0, 0, 0 };
    uint32_t area_ = 0, volume_ = 0;
    std::vector<T> data_;

protected:
    void init(const int3 dims, bool should_clear = false, const T& clear_color = T(0.0f)) {
        dims_ = dims;
        area_ = dims.x() * dims.y();
        volume_ = dims.x() * dims.y() * dims.z();

        data_.resize(volume_);

        if(should_clear) {
            clear(clear_color);
        }
    }

public:
    tensor() = default;

    tensor(const int width, const int height, const int depth) {
        this->init({ width, height, depth });
    }

    tensor(const int width, const int height, const int depth, const T clearColor) {
        this->init({ width, height, depth }, true, clearColor);
    }

    tensor(const int3 dim, const T clearColor) {
        this->init(dim, true, clearColor);
    }

    tensor(const tensor& image) {
        this->init(image.dims_);
        this->copy(image);
    }

    tensor(std::span<const T> input) {
        init({ static_cast<int>(input.size()), 1, 1 });
        std::ranges::copy(input, data_.begin());
    }

    tensor(std::span<const T> input, int width) {
        init({ width, input.size() / width, 1 });
        std::ranges::copy(input, data_.begin());
    }

    tensor(std::span<const float> input) requires (!std::same_as<T, float>) {
        std::span<const T> input2(reinterpret_cast<const T*>(input.data()), input.size() / T::count);
        init({ input2.size(), 1, 1 });
        std::ranges::copy(input2, data_.begin());
    }

    tensor(std::span<const float> input, int width) requires (!std::same_as<T, float>) {
        std::span<const T> input2(reinterpret_cast<const T*>(input.data()), input.size() / T::count);
        init({ width, static_cast<int>(input2.size() / width), 1 });
        std::ranges::copy(input2, data_.begin());
    }

    const T* get_data() const {
        return data_.data();
    }

    inline int index(int x, int y = 0, int z = 0) const {
        return (z * dims_.y() + y) * dims_.x() + x;
    }

    [[nodiscard]] const T& get(int x, int y, int z) const {
        return data_[index(x, y, z)];
    }

    void set(int x, int y, int z, const T& value) {
        data_[index(x, y, z)] = value;
    }

    //void setPixels(const float* pPixels, const int width, const int height) {
    //    init({ width, height, 1 });
    //    memcpy((void*)data_, pPixels, size_t(width) * height * sizeof(T));
    //}

    int3 get_dimensions() const {
        return dims_;
    }

    int get_width() const {
        return dims_.x();
    }

    int get_height() const {
        return dims_.y();
    }

    int get_depth() const {
        return dims_.z();
    }

    void apply_color_map(const tensor<float>& srcImage, const tensor<float3>& colorMap) requires std::same_as<T, float3> {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, colorMap.get(static_cast<int>(srcImage.get(x, y, z) * 255.0f + 0.5f) % colorMap.get_width(), 0, 0));
                }
            }
        }
    }

    void srgb_to_ycxcz() {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, FLIP::xyz_to_ycxcz(FLIP::linear_rgb_to_xyz(FLIP::srgb_to_linear_rgb(get(x, y, z)))));
                }
            }
        }
    }

    void srgb_to_linear_rgb() {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, FLIP::srgb_to_linear_rgb(get(x, y, z)));
                }
            }
        }
    }

    void linear_rgb_to_ycxcz() {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, FLIP::xyz_to_ycxcz(FLIP::linear_rgb_to_xyz(get(x, y, z))));
                }
            }
        }
    }

    void linear_rgb_to_srgb() {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, FLIP::linear_rgb_to_srgb(get(x, y, z)));
                }
            }
        }
    }

    void clear(const T color = T(0.0f)) {
        std::ranges::fill(data_, color);
    }

    void clamp(float low = 0.0f, float high = 1.0f) {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, get(x, y, z).clamp(low, high));
                }
            }
        }
    }

    void apply_tonemap(tonemapper tm) {
        if(tm == tonemapper::reinhard) {
            for(int z = 0; z < get_depth(); z++) {
            #pragma omp parallel for
                for(int y = 0; y < get_height(); y++) {
                    for(int x = 0; x < get_width(); x++) {
                        float3 color = get(x, y, z);
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

    void copy(const tensor<T>& srcImage) {
        if(dims_.x() == srcImage.get_width() && dims_.y() == srcImage.get_height() && dims_.z() == srcImage.get_depth()) {
            data_ = srcImage.data_;
        }
    }

    void expand_grayscale(const tensor<float>& srcImage) {
        for(int z = 0; z < get_depth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < get_height(); y++) {
                for(int x = 0; x < get_width(); x++) {
                    set(x, y, z, float3(srcImage.get(x, y, z)));
                }
            }
        }
    }
};

}
