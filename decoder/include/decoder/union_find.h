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

    struct UFNode {
        int parent;
        int rank;
        int cluster_size;
        bool is_boundary;
    };

    std::vector<UFNode> nodes_;

    struct EdgeState {
        int growth;
        bool fully_grown;
    };

    std::vector<EdgeState> edge_states_;
    int boundary_node_;

    int find(int x);
    void unite(int x, int y);
    bool is_odd_cluster(int x);

    void initialize(const std::vector<bool>& detection_events);
    bool growth_and_merge();
    std::vector<bool> peel();
};

} // namespace decoder
