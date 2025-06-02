//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <cstddef>
#include <utility>
#include <array>
#include <sstream>
#include <stdexcept>
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <iostream>
#include <functional>


namespace utec::algebra {

    template<typename T, std::size_t N>
    class Tensor {
    private:
        std::array<T, N> dimensions_; // La forma del tensor
        std::vector<T> data_;         // Los elementos del tensor

        void validate_dimensions(const std::vector<T>& dims) {
            if (dims.size() != N) {
                std::ostringstream oss;
                oss << "ERROR: Number of dimensions do not match with " << N;
                throw std::invalid_argument(oss.str());
            }
        }

        size_t calculate_total_elements() const {
            if (dimensions_.empty()) return 0;

            size_t total = 1;
            for (const auto& dim : dimensions_) {
                total *= dim;
            }
            return total;
        }

    public:
        Tensor() : dimensions_{}, data_() {}

        //Constructor con dimensiones especificas
        template<typename... Args>
        Tensor(Args&&... args) {
            std::vector<T> dims = { static_cast<T>(std::forward<Args>(args))... };
            validate_dimensions(dims);
            std::copy(dims.begin(), dims.begin() + N, dimensions_.begin());

            data_.resize(calculate_total_elements());
        }

        // Constructor con inicializador de lista
        Tensor(std::initializer_list<T> dims) {
            std::vector<T> dim_vec(dims.begin(), dims.end());
            validate_dimensions(dim_vec);
            std::copy(dim_vec.begin(), dim_vec.begin() + N, dimensions_.begin());
            data_.resize(calculate_total_elements());
        }

        // Constructor de copia
        Tensor& operator=(std::initializer_list<T> elements) {
            // Comprar que el numero de elementos proporcionados coincide con el tamaño del tensor
            if (elements.size() != calculate_total_elements()) {
                std::ostringstream oss;
                oss << "ERROR: Number of elements (" << elements.size() << ") does not match the tensor size (" << calculate_total_elements() << ")";
                throw std::invalid_argument(oss.str());
            }
            data_.assign(elements.begin(), elements.end());
            return *this;
        }

        template<typename... Args>
        void reshape(Args&&... args) {
            std::vector<T> new_dims = { static_cast<T>(std::forward<Args>(args))... };
            validate_dimensions(new_dims);

            // Calcular nuevo numero de elementos a partir de las nuevas dimensiones
            size_t new_size = 1;
            for (const auto& dim : new_dims) {
                new_size *= dim;
            }

            // Comprobar que la nueva forma es compatible con el tamaño actual
            if (new_size != data_.size()) {
                std::ostringstream oss;
                oss << "ERROR: Cannot reshape tensor of size " << data_.size()
                    << " to size " << new_size;
                throw std::invalid_argument(oss.str());
            }
            std::copy(new_dims.begin(), new_dims.begin() + N, dimensions_.begin());
        }

        static constexpr std::size_t size() {
            return N;
        }

        T& operator[](std::size_t idx) {
            return data_[idx];
        }

        const T& operator[](std::size_t idx) const {
            return data_[idx];
        }

        std::array<T, N> shape() const {
            return dimensions_;
        }

        void fill(const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T, N>& tensor) {
        // TODO: implementar la impresión del tensor
            return os;
        }

        bool operator==(const Tensor<T, N>& other) const {
            if (dimensions_ != other.dimensions_) {
                return false;
            }
            return data_ == other.data_;
        }

        bool operator!=(const Tensor<T, N>& other) const {
            return !(*this == other);
        }
    };
}

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
