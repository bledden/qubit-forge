#include "quantum/statevec.h"
#include "quantum/gates.h"
#include <cassert>
#include <cstdio>
#include <cmath>

int main() {
    // Test 1: 3-qubit state vector initialized to |000⟩
    {
        quantum::StateVector sv(3);
        auto amps = sv.download();

        assert(amps.size() == 8);
        assert(std::abs(amps[0].x - 1.0) < 1e-12);
        assert(std::abs(amps[0].y) < 1e-12);
        for (int i = 1; i < 8; i++) {
            assert(std::abs(amps[i].x) < 1e-12);
            assert(std::abs(amps[i].y) < 1e-12);
        }
        printf("PASS: StateVector init_zero\n");
    }

    // Test 2: Apply H to qubit 0 of |000⟩ → (|000⟩ + |001⟩)/√2
    {
        quantum::StateVector sv(3);
        sv.apply_gate_1q(quantum::gates::H(), 0);
        auto amps = sv.download();

        double s = 1.0 / std::sqrt(2.0);
        assert(std::abs(amps[0].x - s) < 1e-10);
        assert(std::abs(amps[1].x - s) < 1e-10);
        for (int i = 2; i < 8; i++) {
            assert(std::abs(amps[i].x) < 1e-10);
            assert(std::abs(amps[i].y) < 1e-10);
        }
        printf("PASS: Hadamard on qubit 0\n");
    }

    // Test 3: Apply H to qubit 1 of |000⟩ → (|000⟩ + |010⟩)/√2
    {
        quantum::StateVector sv(3);
        sv.apply_gate_1q(quantum::gates::H(), 1);
        auto amps = sv.download();

        double s = 1.0 / std::sqrt(2.0);
        assert(std::abs(amps[0].x - s) < 1e-10);
        assert(std::abs(amps[2].x - s) < 1e-10);
        printf("PASS: Hadamard on qubit 1\n");
    }

    // Test 4: HH = I
    {
        quantum::StateVector sv(3);
        sv.apply_gate_1q(quantum::gates::H(), 0);
        sv.apply_gate_1q(quantum::gates::H(), 0);
        auto amps = sv.download();
        assert(std::abs(amps[0].x - 1.0) < 1e-10);
        for (int i = 1; i < 8; i++) {
            assert(std::abs(amps[i].x) < 1e-10);
        }
        printf("PASS: HH = I\n");
    }

    // Test 5: XX = I
    {
        quantum::StateVector sv(2);
        sv.apply_gate_1q(quantum::gates::X(), 0);
        sv.apply_gate_1q(quantum::gates::X(), 0);
        auto amps = sv.download();
        assert(std::abs(amps[0].x - 1.0) < 1e-10);
        printf("PASS: XX = I\n");
    }

    printf("\nAll tests passed!\n");
    return 0;
}
