#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "decoder/types.h"
#include "decoder/union_find.h"

namespace py = pybind11;

PYBIND11_MODULE(pydecoder, m) {
    m.doc() = "QEC decoder library — Union-Find and BP decoders";

    py::class_<decoder::GraphEdge>(m, "GraphEdge")
        .def(py::init<>())
        .def_readwrite("source", &decoder::GraphEdge::source)
        .def_readwrite("target", &decoder::GraphEdge::target)
        .def_readwrite("weight", &decoder::GraphEdge::weight)
        .def_readwrite("error_prob", &decoder::GraphEdge::error_prob)
        .def_readwrite("observable_mask", &decoder::GraphEdge::observable_mask)
    ;

    py::class_<decoder::SyndromeGraph>(m, "SyndromeGraph")
        .def(py::init<>())
        .def_readwrite("n_detectors", &decoder::SyndromeGraph::n_detectors)
        .def_readwrite("n_observables", &decoder::SyndromeGraph::n_observables)
        .def_readwrite("edges", &decoder::SyndromeGraph::edges)
        .def("build_adjacency", &decoder::SyndromeGraph::build_adjacency)
    ;

    py::class_<decoder::DecoderResult>(m, "DecoderResult")
        .def_readonly("observable_prediction", &decoder::DecoderResult::observable_prediction)
        .def_readonly("confidence", &decoder::DecoderResult::confidence)
        .def_readonly("converged", &decoder::DecoderResult::converged)
    ;

    py::class_<decoder::UnionFindDecoder>(m, "UnionFindDecoder")
        .def(py::init<const decoder::SyndromeGraph&>())
        .def("decode", &decoder::UnionFindDecoder::decode)
        .def("decode_batch", [](decoder::UnionFindDecoder& dec,
                                py::array_t<bool> det_events) {
            auto buf = det_events.unchecked<2>();
            int n_shots = buf.shape(0);
            int n_det = buf.shape(1);

            // Get n_observables from first decode
            std::vector<bool> first_det(n_det);
            for (int d = 0; d < n_det; d++) first_det[d] = buf(0, d);
            auto first_result = dec.decode(first_det);
            int n_obs = first_result.observable_prediction.size();

            auto predictions = py::array_t<bool>({n_shots, n_obs});
            auto pred_buf = predictions.mutable_unchecked<2>();

            // Write first result
            for (int o = 0; o < n_obs; o++) {
                pred_buf(0, o) = first_result.observable_prediction[o];
            }

            // Decode remaining
            for (int s = 1; s < n_shots; s++) {
                std::vector<bool> det(n_det);
                for (int d = 0; d < n_det; d++) det[d] = buf(s, d);
                auto result = dec.decode(det);
                for (int o = 0; o < n_obs; o++) {
                    pred_buf(s, o) = result.observable_prediction[o];
                }
            }
            return predictions;
        })
    ;
}
