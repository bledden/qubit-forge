#include "quantum/noise.h"

namespace quantum {

NoiseModel::NoiseModel(uint64_t seed)
    : rng_(seed ? seed : std::random_device{}())
    , uniform_(0.0, 1.0)
    , single_qubit_noise_({NoiseType::Depolarizing, 0.0})
    , two_qubit_noise_({NoiseType::Depolarizing, 0.0})
    , meas_error_prob_(0.0)
    , has_1q_noise_(false)
    , has_2q_noise_(false) {}

void NoiseModel::set_single_qubit_noise(NoiseType type, double probability) {
    single_qubit_noise_ = {type, probability};
    has_1q_noise_ = probability > 0.0;
}

void NoiseModel::set_two_qubit_noise(NoiseType type, double probability) {
    two_qubit_noise_ = {type, probability};
    has_2q_noise_ = probability > 0.0;
}

void NoiseModel::set_measurement_error(double probability) {
    meas_error_prob_ = probability;
}

GateOp NoiseModel::sample_pauli(int target) {
    int choice = std::uniform_int_distribution<int>(0, 2)(rng_);
    GateOp op;
    op.target = target;
    op.control = -1;
    op.param = 0;
    op.custom_1q = {};
    op.custom_2q = {};
    switch (choice) {
        case 0: op.type = GateType::X; break;
        case 1: op.type = GateType::Y; break;
        case 2: op.type = GateType::Z; break;
    }
    return op;
}

std::vector<GateOp> NoiseModel::sample_errors(const GateOp& after_gate) {
    std::vector<GateOp> errors;

    if (after_gate.is_single_qubit() && has_1q_noise_) {
        if (uniform_(rng_) < single_qubit_noise_.probability) {
            switch (single_qubit_noise_.type) {
                case NoiseType::Depolarizing:
                    errors.push_back(sample_pauli(after_gate.target));
                    break;
                case NoiseType::BitFlip: {
                    GateOp op;
                    op.type = GateType::X;
                    op.target = after_gate.target;
                    op.control = -1;
                    op.param = 0;
                    op.custom_1q = {};
                    op.custom_2q = {};
                    errors.push_back(op);
                    break;
                }
                case NoiseType::PhaseFlip: {
                    GateOp op;
                    op.type = GateType::Z;
                    op.target = after_gate.target;
                    op.control = -1;
                    op.param = 0;
                    op.custom_1q = {};
                    op.custom_2q = {};
                    errors.push_back(op);
                    break;
                }
                case NoiseType::AmplitudeDamping: {
                    GateOp op;
                    op.type = GateType::X;
                    op.target = after_gate.target;
                    op.control = -1;
                    op.param = 0;
                    op.custom_1q = {};
                    op.custom_2q = {};
                    errors.push_back(op);
                    break;
                }
            }
        }
    }

    if (!after_gate.is_single_qubit() && has_2q_noise_) {
        // Apply independent noise to each qubit involved
        int qubits[] = {after_gate.target, after_gate.control};
        for (int q : qubits) {
            if (q < 0) continue;
            if (uniform_(rng_) < two_qubit_noise_.probability) {
                switch (two_qubit_noise_.type) {
                    case NoiseType::Depolarizing:
                        errors.push_back(sample_pauli(q));
                        break;
                    case NoiseType::BitFlip: {
                        GateOp op;
                        op.type = GateType::X;
                        op.target = q;
                        op.control = -1;
                        op.param = 0;
                        op.custom_1q = {};
                        op.custom_2q = {};
                        errors.push_back(op);
                        break;
                    }
                    case NoiseType::PhaseFlip: {
                        GateOp op;
                        op.type = GateType::Z;
                        op.target = q;
                        op.control = -1;
                        op.param = 0;
                        op.custom_1q = {};
                        op.custom_2q = {};
                        errors.push_back(op);
                        break;
                    }
                    case NoiseType::AmplitudeDamping: {
                        GateOp op;
                        op.type = GateType::X;
                        op.target = q;
                        op.control = -1;
                        op.param = 0;
                        op.custom_1q = {};
                        op.custom_2q = {};
                        errors.push_back(op);
                        break;
                    }
                }
            }
        }
    }

    return errors;
}

int64_t NoiseModel::apply_measurement_error(int64_t result, int n_qubits) {
    if (meas_error_prob_ <= 0.0) return result;

    int64_t noisy_result = result;
    for (int q = 0; q < n_qubits; q++) {
        if (uniform_(rng_) < meas_error_prob_) {
            noisy_result ^= (1LL << q);  // flip this bit
        }
    }
    return noisy_result;
}

} // namespace quantum
