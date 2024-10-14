#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <iostream>

//g++ -O2 -Ofast -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` app/tensor.cpp -o tensor`python3.12-config --extension-suffix`

namespace py = pybind11;

class Tensor {
public:
    // Constructor que recibe las dimensiones del tensor
    Tensor(const std::vector<size_t>& dimensions) : dims(dimensions) {
        size_t total_size = 1;
        for (size_t dim : dims) {
            total_size *= dim;
        }
        data.resize(total_size, 0.0);  // Inicializamos el tensor con ceros
    }

    // Obtener las dimensiones del tensor
    std::vector<size_t> get_dimensions() const {
        return dims;
    }

    // Establecer un valor en una posición determinada
    void set_value(const std::vector<size_t>& indices, double value) {
        size_t index = compute_index(indices);
        data[index] = value;
    }

    // Obtener el valor de una posición determinada
    double get_value(const std::vector<size_t>& indices) const {
        size_t index = compute_index(indices);
        return data[index];
    }

    std::vector<double> get_tensor() {
        std::vector<double> output(data.size(), 0.0);
        for (size_t i = 0; i < data.size(); ++i) {
            output[i] = data[i];
        }
        return output;
    }
    void print_tensor() const {
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

private:
    std::vector<size_t> dims;  // Dimensiones del tensor
    std::vector<double> data;  // Datos del tensor en formato lineal

    // Método para calcular el índice en el arreglo lineal
    size_t compute_index(const std::vector<size_t>& indices) const {
        size_t index = 0;
        size_t stride = 1;
        for (size_t i = dims.size(); i-- > 0;) {
            index += indices[i] * stride;
            stride *= dims[i];
        }
        return index;
    }
};

// Exponiendo la clase Tensor a Python usando Pybind11
PYBIND11_MODULE(tensor, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>())  // Constructor que toma las dimensiones
        .def("get_dimensions", &Tensor::get_dimensions)
        .def("set_value", &Tensor::set_value)
        .def("get_value", &Tensor::get_value)
        .def("get_tensor", &Tensor::get_tensor);
}
