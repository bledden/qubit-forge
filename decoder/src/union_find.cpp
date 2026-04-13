#include "decoder/union_find.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <numeric>

namespace decoder {

UnionFindDecoder::UnionFindDecoder(const SyndromeGraph& graph)
    : graph_(graph), boundary_node_(graph.n_detectors)
{
    // Pre-compute integer weights from error probabilities.
    // Weight = max(1, ceil(-log(p))) — high-probability errors get low weight
    // (fuse quickly), low-probability errors get high weight (fuse slowly).
    // This is the key to accuracy: clusters connect through likely error paths first.
    edge_weights_.resize(graph_.edges.size());
    for (size_t i = 0; i < graph_.edges.size(); i++) {
        double p = graph_.edges[i].error_prob;
        if (p > 0 && p < 1) {
            edge_weights_[i] = std::max(1, (int)std::ceil(-std::log(p)));
        } else {
            edge_weights_[i] = 1;
        }
    }
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
    for (int i = 0; i < graph_.n_detectors && i < (int)detection_events.size(); i++) {
        if (detection_events[i]) {
            nodes_[i].cluster_size = 1;
        }
    }
    edge_states_.assign(graph_.edges.size(), {0, false});
}

bool UnionFindDecoder::growth_and_merge() {
    bool any_odd = false;

    // Growth: odd clusters grow their boundary edges
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) continue;

        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        bool src_odd = is_odd_cluster(src);
        bool tgt_odd = is_odd_cluster(tgt);

        if (src_odd || tgt_odd) {
            // Both-side growth: each odd endpoint contributes 1 growth unit.
            // An edge with two odd endpoints grows twice as fast.
            int inc = (src_odd ? 1 : 0) + (tgt_odd ? 1 : 0);
            edge_states_[e].growth += inc;
            any_odd = true;
        }

        // Fully grown when growth reaches the weight
        if (edge_states_[e].growth >= edge_weights_[e]) {
            edge_states_[e].fully_grown = true;
        }
    }

    if (!any_odd) return false;

    // Merge: unite endpoints of fully-grown edges
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (!edge_states_[e].fully_grown) continue;
        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;
        unite(src, tgt);
    }

    return true;
}

std::vector<bool> UnionFindDecoder::peel(const std::vector<bool>& detection_events) {
    std::vector<bool> obs_pred(graph_.n_observables, false);
    int n = graph_.n_detectors + 1;

    // Collect fully-grown edges, sorted by weight (low weight = high probability first).
    // This builds a spanning tree that prefers high-probability error paths.
    std::vector<int> grown_edge_indices;
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) {
            grown_edge_indices.push_back(e);
        }
    }
    std::sort(grown_edge_indices.begin(), grown_edge_indices.end(),
              [&](int a, int b) { return edge_weights_[a] < edge_weights_[b]; });

    // Build a cycle-free spanning tree using union-find
    std::vector<int> tree_parent(n);
    std::iota(tree_parent.begin(), tree_parent.end(), 0);
    std::vector<int> tree_rank(n, 0);

    auto tree_find = [&](int x) {
        while (tree_parent[x] != x) {
            tree_parent[x] = tree_parent[tree_parent[x]];
            x = tree_parent[x];
        }
        return x;
    };

    std::vector<std::vector<std::pair<int, int>>> adj(n);  // (neighbor, edge_idx)
    std::vector<int> degree(n, 0);

    for (int e_idx : grown_edge_indices) {
        const auto& edge = graph_.edges[e_idx];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        int rs = tree_find(src), rt = tree_find(tgt);
        if (rs != rt) {
            // Add to spanning tree (no cycle)
            if (tree_rank[rs] < tree_rank[rt]) std::swap(rs, rt);
            tree_parent[rt] = rs;
            if (tree_rank[rs] == tree_rank[rt]) tree_rank[rs]++;

            adj[src].push_back({tgt, e_idx});
            adj[tgt].push_back({src, e_idx});
            degree[src]++;
            degree[tgt]++;
        }
    }

    // Peel from leaves: if leaf subtree has odd defect parity, include the edge
    std::queue<int> leaves;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) leaves.push(i);
    }

    std::vector<bool> node_done(n, false);
    std::vector<bool> edge_done(graph_.edges.size(), false);

    // Use ORIGINAL detection events for parity (not merged cluster sizes)
    std::vector<int> defect_parity(n, 0);
    for (int i = 0; i < graph_.n_detectors && i < (int)detection_events.size(); i++) {
        if (detection_events[i]) {
            defect_parity[i] = 1;
        }
    }

    while (!leaves.empty()) {
        int leaf = leaves.front();
        leaves.pop();
        if (node_done[leaf]) continue;
        node_done[leaf] = true;

        for (auto& [nbr, e_idx] : adj[leaf]) {
            if (edge_done[e_idx] || node_done[nbr]) continue;
            edge_done[e_idx] = true;

            if (defect_parity[leaf] % 2 == 1) {
                // This edge is part of the correction
                const auto& edge = graph_.edges[e_idx];
                for (int obs : edge.observable_mask) {
                    if (obs >= 0 && obs < (int)obs_pred.size()) {
                        obs_pred[obs] = !obs_pred[obs];
                    }
                }
                // Propagate parity to neighbor
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

    // Growth + merge until no odd clusters remain
    int max_rounds = 0;
    for (int w : edge_weights_) max_rounds = std::max(max_rounds, w);
    max_rounds += 10;

    for (int round = 0; round < max_rounds; round++) {
        if (!growth_and_merge()) break;
    }

    auto prediction = peel(detection_events);

    DecoderResult result;
    result.observable_prediction = prediction;
    result.confidence = 1.0;
    result.converged = true;
    return result;
}

} // namespace decoder
