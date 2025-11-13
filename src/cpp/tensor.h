#pragma once
#include <vector>
#include "vecs.h"

namespace FLIP {

template<typename T>
class tensor {
protected:
    std::vector<T> data_;
    int3 dims_ = { 0, 0, 0 };
    uint32_t area_ = 0, volume_ = 0;

    void init(const int3& dims, bool should_clear = false, const T& clear_color = T(0.0f));

public:
    tensor() = default;

    tensor(const int width, const int height, const int depth) {
        init({ width, height, depth });
    }

    tensor(const int width, const int height, const int depth, const T clearColor) {
        init({ width, height, depth }, true, clearColor);
    }

    tensor(const int3 dim, const T clearColor) {
        init(dim, true, clearColor);
    }

    tensor(const tensor& image) {
        init(image.dims_);
        copy(image);
    }

    tensor(tensor&& image) noexcept
        : data_(std::move(image.data_)), dims_(image.dims_), area_(image.area_), volume_(image.volume_) {}

    tensor(std::span<const T> input) {
        init({ static_cast<int>(input.size()), 1, 1 });
        std::ranges::copy(input, data_.begin());
    }

    tensor(std::span<const T> input, int width) {
        init({ width, static_cast<int>(input.size()) / width, 1 });
        std::ranges::copy(input, data_.begin());
    }

    tensor(std::span<const float> input) requires (!std::same_as<T, float>) {
        std::span<const T> input2(reinterpret_cast<const T*>(input.data()), input.size() / T::count);
        init({ static_cast<int>(input2.size()), 1, 1 });
        std::ranges::copy(input2, data_.begin());
    }

    tensor(std::span<const float> input, int width) requires (!std::same_as<T, float>) {
        std::span<const T> input2(reinterpret_cast<const T*>(input.data()), input.size() / T::count);
        init({ width, static_cast<int>(input2.size() / width), 1 });
        std::ranges::copy(input2, data_.begin());
    }

    tensor& operator=(const tensor&) = default;
    tensor& operator=(tensor&&) = default;

    const T* get_data() const {
        return data_.data();
    }

    [[nodiscard]] int index(int x, int y = 0, int z = 0) const {
        return (z * dims_.y() + y) * dims_.x() + x;
    }

    [[nodiscard]] const T& get(int x, int y, int z) const {
        return data_[index(x, y, z)];
    }

    void set(int x, int y, int z, const T& value) {
        data_[index(x, y, z)] = value;
    }

    [[nodiscard]] int3 get_dimensions() const {
        return dims_;
    }

    [[nodiscard]] int get_width() const {
        return dims_.x();
    }

    [[nodiscard]] int get_height() const {
        return dims_.y();
    }

    [[nodiscard]] int get_depth() const {
        return dims_.z();
    }

    void srgb_to_ycxcz() requires floatN_at_least<T, 3>;
    void srgb_to_linear_rgb() requires floatN_at_least<T, 3>;
    void linear_rgb_to_ycxcz() requires floatN_at_least<T, 3>;
    void linear_rgb_to_srgb() requires floatN_at_least<T, 3>;
    void apply_tonemap(tonemapper tm) requires floatN_at_least<T, 3>;
    void apply_color_map(const tensor<float>& srcImage, const tensor<float3>& colorMap) requires floatN_at_least<T, 3>;
    void expand_grayscale(const tensor<float>& srcImage);

    void clamp(float low = 0.0f, float high = 1.0f);

    void clear(const T color = T(0.0f)) {
        std::ranges::fill(data_, color);
    }

    void copy(const tensor& srcImage) {
        if(dims_.x() == srcImage.get_width() && dims_.y() == srcImage.get_height() && dims_.z() == srcImage.get_depth()) {
            data_ = srcImage.data_;
        }
    }

};


}
