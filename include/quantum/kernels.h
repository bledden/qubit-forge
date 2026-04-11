#pragma once

#include "quantum/types.h"
#include <hip/hip_runtime.h>

namespace quantum {

void launch_apply_1q(Complex128* state, int n_qubits, int target, const Gate1Q& gate,
                     hipStream_t stream = 0);

void launch_apply_2q(Complex128* state, int n_qubits, int qubit_a, int qubit_b,
                     const Gate2Q& gate, hipStream_t stream = 0);

} // namespace quantum
