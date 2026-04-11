#pragma once

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace quantum {

struct Complex128 {
    double x, y;

    __host__ __device__ Complex128() : x(0.0), y(0.0) {}
    __host__ __device__ Complex128(double r, double i = 0.0) : x(r), y(i) {}

    __host__ __device__ Complex128 operator+(const Complex128& o) const {
        return {x + o.x, y + o.y};
    }
    __host__ __device__ Complex128 operator-(const Complex128& o) const {
        return {x - o.x, y - o.y};
    }
    __host__ __device__ Complex128 operator*(const Complex128& o) const {
        return {x * o.x - y * o.y, x * o.y + y * o.x};
    }
    __host__ __device__ Complex128 operator*(double s) const {
        return {x * s, y * s};
    }
    __host__ __device__ double norm2() const { return x * x + y * y; }
};

struct Gate1Q {
    Complex128 m[2][2];
};

struct Gate2Q {
    Complex128 m[4][4];
};

} // namespace quantum
