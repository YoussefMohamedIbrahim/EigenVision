#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <linalg/Matrix.hpp>
#include "PCA.hpp"
#include "KNN.hpp"
#include "DataLoader.hpp"

namespace py = pybind11;
using namespace linalg;

// We only bind Matrix<double> because that's what PCA/KNN use
using Mat = Matrix<double>;

PYBIND11_MODULE(eigenvision, m)
{
    m.doc() = "EigenVision: C++ Accelerated PCA & KNN Library";

    py::class_<Mat>(m, "Matrix")
        .def(py::init<>()) // Default constructor
        .def(py::init<size_t, size_t>())
        .def("rows", &Mat::rows)
        .def("cols", &Mat::cols)
        // Bind operator() for easy access: mat[row, col]
        .def("__getitem__", [](const Mat &m, std::pair<size_t, size_t> idx)
             { return m(idx.first, idx.second); })
        .def("__setitem__", [](Mat &m, std::pair<size_t, size_t> idx, double val)
             { m(idx.first, idx.second) = val; });

    py::class_<PCA>(m, "PCA")
        .def(py::init<>())
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        .def("save", &PCA::save)
        .def("load", &PCA::load);

    py::class_<KNN>(m, "KNN")
        .def(py::init<>())
        .def("fit", &KNN::fit)
        .def("predict", &KNN::predict, py::arg("query"), py::arg("k") = 5)
        .def("evaluate", &KNN::evaluate, py::arg("test_features"), py::arg("test_labels"), py::arg("k") = 5)
        .def("save", &KNN::save)
        .def("load", &KNN::load);

    py::class_<DataSet>(m, "DataSet")
        .def_readwrite("images", &DataSet::images)
        .def_readwrite("labels", &DataSet::labels);

    py::class_<DataLoader>(m, "DataLoader")
        .def_static("load", &DataLoader::load);
}