#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

#include "embedder.h"
#include "tokenizer_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(_static, m) {
    m.doc() = "Static Python inference pipeline";

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def(py::init<Tokenizer::Model>())
        .def(py::init<const std::string&>())
        .def("encode", &Tokenizer::encode)
        .def("encode_ordinary", &Tokenizer::encode_ordinary)
        .def("decode", &Tokenizer::decode)
        .def_property_readonly("vocab_size", &Tokenizer::vocab_size);

    py::class_<BatchTokenizer>(m, "BatchTokenizer")
        .def(py::init<>())
        .def(py::init<Tokenizer::Model>())
        .def(py::init<const std::string&>())
        .def("encode", &BatchTokenizer::encode);

    py::class_<Embedder>(m, "Embedder")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<uint32_t, const std::string&>())
        .def("load_embeddings", &Embedder::load_embeddings, py::arg("path") = "")
        .def("save_binary", &Embedder::save_binary, py::arg("path") = "", py::arg("embedding_dim") = Embedder::DEFAULT_EMBEDDING_DIM)
        .def("load_binary", &Embedder::load_binary, py::arg("path") = "", py::arg("max_token_id") = 0)
        .def("get_single_embedding", [](Embedder& self, const std::string& text) {
            auto emb = self.get_single_embedding(text);
            return py::array_t<float, py::array::c_style | py::array::forcecast>(emb.size(), emb.data());
        })
        .def("get_token_embeddings", [](Embedder& self, const std::vector<std::string>& texts) {
            auto emb = self.get_token_embeddings(texts);
            py::list result;
            for (const auto& e : emb) {
                result.append(py::array_t<float, py::array::c_style | py::array::forcecast>(e.size(), e.data()));
            }
            return result;
        })
        .def("get_embedding_from_tokens", [](Embedder& self, const std::vector<uint32_t>& tokens) {
            auto emb = self.get_embedding_from_tokens(tokens);
            return py::array_t<float, py::array::c_style | py::array::forcecast>(emb.size(), emb.data());
        })
        .def("get_embeddings_from_token_batches", [](Embedder& self, const std::vector<std::vector<uint32_t>>& token_batches) {
            auto emb = self.get_embeddings_from_token_batches(token_batches);
            py::list result;
            for (const auto& e : emb) {
                result.append(py::array_t<float, py::array::c_style | py::array::forcecast>(e.size(), e.data()));
            }
            return result;
        })
        .def("set_tokenizer", &Embedder::set_tokenizer)
        .def_property_readonly("embedding_dim", &Embedder::embedding_dim);

    py::enum_<Tokenizer::Model>(m, "TokenizerModel")
        .value("R50K_BASE", Tokenizer::Model::R50K_BASE)
        .value("P50K_BASE", Tokenizer::Model::P50K_BASE)
        .value("P50K_EDIT", Tokenizer::Model::P50K_EDIT)
        .value("CL100K_BASE", Tokenizer::Model::CL100K_BASE)
        .value("O200K_BASE", Tokenizer::Model::O200K_BASE)
        .value("O200K_HARMONY", Tokenizer::Model::O200K_HARMONY)
        .export_values();
}
