#include "quantum/fusion.h"
#include <unordered_set>

namespace quantum {

Gate1Q multiply_gates(const Gate1Q& a, const Gate1Q& b) {
    Gate1Q r;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j];
        }
    }
    return r;
}

static std::vector<GateOp> fuse_same_qubit(const std::vector<GateOp>& ops) {
    std::vector<GateOp> result;

    for (size_t i = 0; i < ops.size(); i++) {
        if (!ops[i].is_single_qubit()) {
            result.push_back(ops[i]);
            continue;
        }

        Gate1Q fused = ops[i].to_gate1q();
        int target = ops[i].target;
        size_t j = i + 1;
        while (j < ops.size() && ops[j].is_single_qubit() && ops[j].target == target) {
            fused = multiply_gates(ops[j].to_gate1q(), fused);
            j++;
        }

        if (j > i + 1) {
            GateOp fused_op;
            fused_op.type = GateType::Custom1Q;
            fused_op.target = target;
            fused_op.control = -1;
            fused_op.param = 0;
            fused_op.custom_1q = fused;
            fused_op.custom_2q = {};
            result.push_back(fused_op);
        } else {
            result.push_back(ops[i]);
        }
        i = j - 1;
    }
    return result;
}

static std::vector<FusedLayer> extract_layers(const std::vector<GateOp>& ops) {
    std::vector<FusedLayer> layers;
    FusedLayer current;
    std::unordered_set<int> used_qubits;

    for (const auto& op : ops) {
        bool conflict = used_qubits.count(op.target);
        if (!conflict && op.control >= 0) {
            conflict = used_qubits.count(op.control);
        }

        if (conflict) {
            layers.push_back(std::move(current));
            current = FusedLayer();
            used_qubits.clear();
        }

        current.ops.push_back(op);
        used_qubits.insert(op.target);
        if (op.control >= 0) {
            used_qubits.insert(op.control);
        }
    }

    if (!current.ops.empty()) {
        layers.push_back(std::move(current));
    }

    return layers;
}

std::vector<FusedLayer> fuse_circuit(const Circuit& circuit) {
    auto fused_ops = fuse_same_qubit(circuit.ops());
    return extract_layers(fused_ops);
}

} // namespace quantum
