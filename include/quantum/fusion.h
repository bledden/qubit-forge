#pragma once

#include "quantum/circuit.h"
#include <vector>

namespace quantum {

struct FusedLayer {
    std::vector<GateOp> ops;
};

std::vector<FusedLayer> fuse_circuit(const Circuit& circuit);
Gate1Q multiply_gates(const Gate1Q& a, const Gate1Q& b);

} // namespace quantum
