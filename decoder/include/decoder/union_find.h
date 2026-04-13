#pragma once

#include "decoder/types.h"
#include <vector>

namespace decoder {

class UnionFindDecoder {
public:
    UnionFindDecoder(const SyndromeGraph& graph);
    DecoderResult decode(const std::vector<bool>& detection_events);

private:
    SyndromeGraph graph_;
    std::vector<int> edge_weights_;  // Pre-computed integer weights from error_prob

    // Union-Find forest (re-initialized per decode call)
    struct UFNode {
        int parent;
        int rank;
        int cluster_size;  // Number of defects in this cluster
        bool is_boundary;
    };

    std::vector<UFNode> nodes_;

    struct EdgeState {
        int growth;        // Total growth from both sides
        bool fully_grown;
    };

    std::vector<EdgeState> edge_states_;
    int boundary_node_;

    // Core operations
    int find(int x);
    void unite(int x, int y);
    bool is_odd_cluster(int x);

    // Decoder phases
    void initialize(const std::vector<bool>& detection_events);
    bool growth_and_merge();
    std::vector<bool> peel(const std::vector<bool>& detection_events);
};

} // namespace decoder
