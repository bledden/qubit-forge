#pragma once

#include "quantum/types.h"
#include <cmath>

namespace quantum {
namespace gates {

inline Gate1Q H() {
    double s = 1.0 / std::sqrt(2.0);
    return {{{
        {Complex128(s, 0), Complex128(s, 0)},
        {Complex128(s, 0), Complex128(-s, 0)}
    }}};
}

inline Gate1Q X() {
    return {{{
        {Complex128(0), Complex128(1)},
        {Complex128(1), Complex128(0)}
    }}};
}

inline Gate1Q Y() {
    return {{{
        {Complex128(0), Complex128(0, -1)},
        {Complex128(0, 1), Complex128(0)}
    }}};
}

inline Gate1Q Z() {
    return {{{
        {Complex128(1), Complex128(0)},
        {Complex128(0), Complex128(-1)}
    }}};
}

inline Gate1Q Rx(double theta) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    return {{{
        {Complex128(c, 0), Complex128(0, -s)},
        {Complex128(0, -s), Complex128(c, 0)}
    }}};
}

inline Gate1Q Ry(double theta) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    return {{{
        {Complex128(c, 0), Complex128(-s, 0)},
        {Complex128(s, 0), Complex128(c, 0)}
    }}};
}

inline Gate1Q Rz(double theta) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    return {{{
        {Complex128(c, -s), Complex128(0)},
        {Complex128(0), Complex128(c, s)}
    }}};
}

inline Gate1Q S() {
    return {{{
        {Complex128(1), Complex128(0)},
        {Complex128(0), Complex128(0, 1)}
    }}};
}

inline Gate1Q T() {
    double s = 1.0 / std::sqrt(2.0);
    return {{{
        {Complex128(1), Complex128(0)},
        {Complex128(0), Complex128(s, s)}
    }}};
}

inline Gate2Q CNOT() {
    Gate2Q g = {};
    g.m[0][0] = Complex128(1);
    g.m[1][1] = Complex128(1);
    g.m[2][3] = Complex128(1);
    g.m[3][2] = Complex128(1);
    return g;
}

inline Gate2Q CZ() {
    Gate2Q g = {};
    g.m[0][0] = Complex128(1);
    g.m[1][1] = Complex128(1);
    g.m[2][2] = Complex128(1);
    g.m[3][3] = Complex128(-1);
    return g;
}

inline Gate2Q SWAP() {
    Gate2Q g = {};
    g.m[0][0] = Complex128(1);
    g.m[1][2] = Complex128(1);
    g.m[2][1] = Complex128(1);
    g.m[3][3] = Complex128(1);
    return g;
}

} // namespace gates
} // namespace quantum
