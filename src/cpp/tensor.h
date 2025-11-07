#pragma once
#include "vecs.h"

namespace FLIP {

template<typename T>
class tensor {
protected:
    int3 mDim = { 0, 0, 0 };
    int mArea = 0, mVolume = 0;
    T* mvpHostData = nullptr;
protected:
    bool allocateHost() {
        this->mvpHostData = (T*)malloc(this->mVolume * sizeof(T));

        if(this->mvpHostData == nullptr) {
            return false;
        }

        return true;
    }

    void init(const int3 dim, bool bClear = false, T clearColor = T(0.0f)) {
        this->mDim = dim;
        this->mArea = dim.x() * dim.y();
        this->mVolume = dim.x() * dim.y() * dim.z();

        allocateHost();

        if(bClear) {
            this->clear(clearColor);
        }
    }

public:
    tensor() { }

    tensor(const int width, const int height, const int depth) {
        this->init({ width, height, depth });
    }

    tensor(const int width, const int height, const int depth, const T clearColor) {
        this->init({ width, height, depth }, true, clearColor);
    }

    tensor(const int3 dim, const T clearColor) {
        this->init(dim, true, clearColor);
    }

    tensor(tensor& image) {
        this->init(image.mDim);
        this->copy(image);
    }

    tensor(const float3* pColorMap, int size) {
        this->init({ size, 1, 1 });
        if(this->mvpHostData != nullptr) {
            memcpy((void*)this->mvpHostData, pColorMap, size * sizeof(float3));
        }
    }
    ~tensor() {
        free(this->mvpHostData);
    }

    T* getHostData() {
        return this->mvpHostData;
    }

    inline int index(int x, int y = 0, int z = 0) const {
        return (z * this->mDim.y() + y) * mDim.x() + x;
    }

    const T& get(int x, int y, int z) const {
        return this->mvpHostData[this->index(x, y, z)];
    }

    void set(int x, int y, int z, const T& value) {
        this->mvpHostData[this->index(x, y, z)] = value;
    }

    void setPixels(const float* pPixels, const int width, const int height) {
        this->init({ width, height, 1 });
        memcpy((void*)this->mvpHostData, pPixels, size_t(width) * height * sizeof(T));
    }

    int3 getDimensions() const {
        return this->mDim;
    }

    int getWidth() const {
        return this->mDim.x();
    }

    int getHeight() const {
        return this->mDim.y();
    }

    int getDepth() const {
        return this->mDim.z();
    }

    void colorMap(const tensor<float>& srcImage, tensor<float3>& colorMap) requires std::same_as<T, float3> {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, colorMap.get(int(srcImage.get(x, y, z) * 255.0f + 0.5f) % colorMap.getWidth(), 0, 0));
                }
            }
        }
    }

    void sRGBToYCxCz() {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, XYZToYCxCz(LinearRGBToXYZ(sRGBToLinearRGB(this->get(x, y, z)))));
                }
            }
        }
    }

    void sRGBToLinearRGB() {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, FLIP::sRGBToLinearRGB(this->get(x, y, z)));
                }
            }
        }
    }

    void LinearRGBToYCxCz() {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, FLIP::XYZToYCxCz(FLIP::LinearRGBToXYZ(this->get(x, y, z))));
                }
            }
        }
    }

    void LinearRGBTosRGB() {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, FLIP::LinearRGBTosRGB(this->get(x, y, z)));
                }
            }
        }
    }

    void clear(const T color = T(0.0f)) {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, color);
                }
            }
        }
    }

    void clamp(float low = 0.0f, float high = 1.0f) {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, float3::clamp(this->get(x, y, z), low, high));
                }
            }
        }
    }

    void toneMap(std::string tm) {
        int toneMapper = 1;
        if(tm == "reinhard") {
            for(int z = 0; z < this->getDepth(); z++) {
            #pragma omp parallel for
                for(int y = 0; y < this->getHeight(); y++) {
                    for(int x = 0; x < this->getWidth(); x++) {
                        float3 color = this->get(x, y, z);
                        float luminance = linearRGBToLuminance(color);
                        float factor = 1.0f / (1.0f + luminance);
                        this->set(x, y, z, color * factor);
                    }
                }
            }
            return;
        }

        if(tm == "aces")
            toneMapper = 1;
        if(tm == "hable")
            toneMapper = 2;

        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    const float* tc = ToneMappingCoefficients[toneMapper];
                    float3 color = this->get(x, y, z);
                    this->set(x, y, z, float3(((color * color) * tc[0] + color * tc[1] + tc[2]) / (color * color * tc[3] + color * tc[4] + tc[5])));
                }
            }
        }
    }

    void copy(tensor<T>& srcImage) {
        if(this->mDim.x() == srcImage.getWidth() && this->mDim.y() == srcImage.getHeight() && this->mDim.z() == srcImage.getDepth()) {
            memcpy((void*)this->mvpHostData, srcImage.getHostData(), this->mVolume * sizeof(T));
        }
    }

    void copyFloatToColor3(tensor<float>& srcImage) {
        for(int z = 0; z < this->getDepth(); z++) {
        #pragma omp parallel for
            for(int y = 0; y < this->getHeight(); y++) {
                for(int x = 0; x < this->getWidth(); x++) {
                    this->set(x, y, z, float3(srcImage.get(x, y, z)));
                }
            }
        }
    }
};

}
