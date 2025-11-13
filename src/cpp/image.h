#pragma once

#include "tensor.h"

namespace FLIP {

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

    image(image&& image) noexcept
        : tensor<T>(std::move(image)) { }

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

    image& operator=(const image&) = default;
    image& operator=(image&&) = default;

    const T& get(int x, int y) const {
        return this->data_[this->index(x, y)];
    }

    void set(int x, int y, const T& value) {
        this->data_[this->index(x, y)] = value;
    }
};

}
