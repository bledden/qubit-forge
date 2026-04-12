#pragma once

#include "quantum/types.h"
#include "quantum/gates.h"
#include "quantum/circuit.h"
#include <vector>
#include <random>

namespace quantum {

// Noise channel types
enum class NoiseType {
    Depolarizing,    // Random X, Y, or Z with probability p
    BitFlip,         // X with probability p
    PhaseFlip,       // Z with probability p
    AmplitudeDamping // |1⟩ → |0⟩ decay (simplified as bit flip for state vector)
};

// A noise channel applied after specific gate types
struct NoiseChannel {
    NoiseType type;
    double probability;  // error probability per gate
};

class NoiseModel {
public:
    NoiseModel(uint64_t seed = 0);

    // Set noise on all single-qubit gates
    void set_single_qubit_noise(NoiseType type, double probability);

    // Set noise on all two-qubit gates
    void set_two_qubit_noise(NoiseType type, double probability);

    // Set measurement error probability
    void set_measurement_error(double probability);

    // Generate error gates to insert after a given gate operation
    // Returns empty vector if no error occurs (based on RNG)
    std::vector<GateOp> sample_errors(const GateOp& after_gate);

    // Apply measurement error to a sampled result
    int64_t apply_measurement_error(int64_t result, int n_qubits);

    double measurement_error_prob() const { return meas_error_prob_; }

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;

    NoiseChannel single_qubit_noise_;
    NoiseChannel two_qubit_noise_;
    double meas_error_prob_;
    bool has_1q_noise_;
    bool has_2q_noise_;

    // Sample a random Pauli gate for a given qubit
    GateOp sample_pauli(int target);
};

} // namespace quantum
