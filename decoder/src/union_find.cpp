#include "decoder/union_find.h"
#include <algorithm>
#include <queue>
#include <cmath>

namespace decoder {

UnionFindDecoder::UnionFindDecoder(const SyndromeGraph& graph)
    : graph_(graph), boundary_node_(graph.n_detectors)
{
    graph_.build_adjacency();
}

int UnionFindDecoder::find(int x) {
    while (nodes_[x].parent != x) {
        nodes_[x].parent = nodes_[nodes_[x].parent].parent;
        x = nodes_[x].parent;
    }
    return x;
}

void UnionFindDecoder::unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx == ry) return;
    if (nodes_[rx].rank < nodes_[ry].rank) std::swap(rx, ry);
    nodes_[ry].parent = rx;
    nodes_[rx].cluster_size += nodes_[ry].cluster_size;
    nodes_[rx].is_boundary = nodes_[rx].is_boundary || nodes_[ry].is_boundary;
    if (nodes_[rx].rank == nodes_[ry].rank) nodes_[rx].rank++;
}

bool UnionFindDecoder::is_odd_cluster(int x) {
    int root = find(x);
    return (nodes_[root].cluster_size % 2 == 1) && !nodes_[root].is_boundary;
}

void UnionFindDecoder::initialize(const std::vector<bool>& detection_events) {
    int n = graph_.n_detectors + 1;
    nodes_.resize(n);
    for (int i = 0; i < n; i++) {
        nodes_[i] = {i, 0, 0, (i == boundary_node_)};
    }
    for (int i = 0; i < graph_.n_detectors; i++) {
        if (i < (int)detection_events.size() && detection_events[i]) {
            nodes_[i].cluster_size = 1;
        }
    }
    edge_states_.assign(graph_.edges.size(), {0, false});
}

bool UnionFindDecoder::growth_and_merge() {
    bool any_odd = false;
    bool any_merge = false;

    // Growth: increment growth on edges touching odd clusters
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) continue;
        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        bool src_odd = is_odd_cluster(src);
        bool tgt_odd = is_odd_cluster(tgt);

        if (src_odd || tgt_odd) {
            edge_states_[e].growth++;
            any_odd = true;
        }

        // Weight: use 1 for unweighted (simpler, still correct for uniform noise)
        if (edge_states_[e].growth >= 1) {
            edge_states_[e].fully_grown = true;
        }
    }

    // Merge: unite endpoints of fully-grown edges
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (!edge_states_[e].fully_grown) continue;
        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;
        if (find(src) != find(tgt)) {
            unite(src, tgt);
            any_merge = true;
        }
    }

    return any_odd;
}

std::vector<bool> UnionFindDecoder::peel() {
    std::vector<bool> obs_pred(graph_.n_observables, false);
    int n = graph_.n_detectors + 1;

    // Build spanning forest of fully-grown edges
    std::vector<std::vector<std::pair<int, int>>> tree(n);
    std::vector<int> degree(n, 0);

    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (!edge_states_[e].fully_grown) continue;
        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;
        tree[src].push_back({tgt, (int)e});
        tree[tgt].push_back({src, (int)e});
        degree[src]++;
        degree[tgt]++;
    }

    // Peel from leaves
    std::queue<int> leaves;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) leaves.push(i);
    }

    std::vector<bool> node_done(n, false);
    std::vector<bool> edge_done(graph_.edges.size(), false);
    std::vector<int> defect_parity(n, 0);
    for (int i = 0; i < graph_.n_detectors; i++) {
        defect_parity[i] = nodes_[i].cluster_size > 0 ? 1 : 0;
    }

    while (!leaves.empty()) {
        int leaf = leaves.front();
        leaves.pop();
        if (node_done[leaf]) continue;
        node_done[leaf] = true;

        for (auto& [nbr, e_idx] : tree[leaf]) {
            if (edge_done[e_idx] || node_done[nbr]) continue;
            edge_done[e_idx] = true;

            if (defect_parity[leaf] % 2 == 1) {
                const auto& edge = graph_.edges[e_idx];
                for (int obs : edge.observable_mask) {
                    if (obs >= 0 && obs < (int)obs_pred.size())
                        obs_pred[obs] = !obs_pred[obs];
                }
                defect_parity[nbr] += defect_parity[leaf];
            }

            degree[nbr]--;
            if (degree[nbr] == 1) leaves.push(nbr);
        }
    }

    return obs_pred;
}

DecoderResult UnionFindDecoder::decode(const std::vector<bool>& detection_events) {
    initialize(detection_events);

    for (int round = 0; round < graph_.n_detectors + 10; round++) {
        if (!growth_and_merge()) break;
    }

    auto prediction = peel();

    DecoderResult result;
    result.observable_prediction = prediction;
    result.confidence = 1.0;
    result.converged = true;
    return result;
}

} // namespace decoder
