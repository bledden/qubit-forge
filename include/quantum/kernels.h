#pragma once

#include "quantum/types.h"
#include <hip/hip_runtime.h>

namespace quantum {

void launch_apply_1q(Complex128* state, int n_qubits, int target, const Gate1Q& gate,
                     hipStream_t stream = 0);

void launch_apply_2q(Complex128* state, int n_qubits, int qubit_a, int qubit_b,
                     const Gate2Q& gate, hipStream_t stream = 0);

void launch_apply_diagonal(Complex128* state, int n_qubits, int target, const Gate1Q& gate,
                           hipStream_t stream = 0);

bool gate_is_diagonal(const Gate1Q& gate);

void launch_probabilities(const Complex128* state, double* probs, int64_t n,
                          hipStream_t stream = 0);
double launch_total_probability(const double* probs, int64_t n, hipStream_t stream = 0);

} // namespace quantum
