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
#include <numeric>

namespace utec::algebra {

    template<typename T, std::size_t N>
    class Tensor {
    private:
        std::array<std::size_t, N> dimensions_; // La forma del tensor
        std::vector<T> data_;         // Los elementos del tensor

        void validate_dimensions(const std::vector<std::size_t>& dims) {
            if (dims.size() != N) {
                std::ostringstream oss;
                oss << "Number of dimensions do not match with " << N;
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

        size_t calculate_linear_index(const std::array<std::size_t, N>& indices) const {
            size_t linear_idx = 0;
            size_t stride = 1;
            for (int i = N - 1; i >= 0; --i) {
                linear_idx += indices[i] * stride;
                stride *= dimensions_[i];
            }
            return linear_idx;
        }

        template <typename OStream>
        void print_tensor_recursive(OStream& os, std::array<std::size_t, N> indices, size_t dim_idx) const {
            if (dim_idx == N - 1) {
                for (size_t i = 0; i < dimensions_[dim_idx]; ++i) {
                    indices[dim_idx] = i;
                    os << data_[calculate_linear_index(indices)];
                    if (i < dimensions_[dim_idx] - 1) {
                        os << " ";
                    }
                }
                return;
            }
            os << "{" << "\n";
            for (size_t i = 0; i < dimensions_[dim_idx]; ++i) {
                indices[dim_idx] = i;
                print_tensor_recursive(os, indices, dim_idx + 1);

                if (i < dimensions_[dim_idx] - 1) {
                    os << "\n";
                }
            }

            os << "\n" << "}";
        }

        bool are_broadcastable(const std::array<std::size_t, N>& shape1,
                              const std::array<std::size_t, N>& shape2) const {
            for (int i = N - 1; i >= 0; --i) {
                if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        std::array<std::size_t, N> get_broadcasted_index(
            const std::array<std::size_t, N>& target_index,
            const std::array<std::size_t, N>& tensor_shape) const {

            std::array<std::size_t, N> broadcasted_index;
            for (size_t i = 0; i < N; ++i) {
                if (tensor_shape[i] == 1) {
                    broadcasted_index[i] = 0;
                } else {
                    broadcasted_index[i] = target_index[i];
                }
            }
            return broadcasted_index;
        }

        std::array<std::size_t, N> get_broadcast_result_shape(
            const std::array<std::size_t, N>& shape1,
            const std::array<std::size_t, N>& shape2) const {

            std::array<std::size_t, N> result_shape;
            for (size_t i = 0; i < N; ++i) {
                result_shape[i] = std::max(shape1[i], shape2[i]);
            }
            return result_shape;
        }

    public:

        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        Tensor() : dimensions_{}, data_() {}

        // Constructor con dimensiones especificas (solo para tipos aritmeticos)
        template<typename... Args>
        Tensor(Args&&... args)
            requires (sizeof...(args) > 0 && (std::is_arithmetic_v<std::decay_t<Args>> && ...))
        {
            std::vector<std::size_t> dims = { static_cast<std::size_t>(std::forward<Args>(args))... };
            validate_dimensions(dims);
            std::copy(dims.begin(), dims.begin() + N, dimensions_.begin());
            data_.resize(calculate_total_elements());
        }

        // Constructor con inicializador de lista
        Tensor(std::initializer_list<std::size_t> dims) {
            std::vector<std::size_t> dim_vec(dims.begin(), dims.end());
            validate_dimensions(dim_vec);
            std::copy(dim_vec.begin(), dim_vec.begin() + N, dimensions_.begin());
            data_.resize(calculate_total_elements());
        }

        // Constructor con array de dimensiones (EXPLÍCITO)
        explicit Tensor(const std::array<std::size_t, N>& dims) {
            dimensions_ = dims;
            data_.resize(calculate_total_elements());
        }

        // Constructor de copia
        Tensor& operator=(std::initializer_list<T> elements) {
            // Comprobar que el numero de elementos proporcionados coincide con el tamaño del tensor
            if (elements.size() != calculate_total_elements()) {
                std::ostringstream oss;
                oss << "Data size does not match tensor size";
                throw std::invalid_argument(oss.str());
            }
            data_.assign(elements.begin(), elements.end());
            return *this;
        }

        template<typename... Args>
        T& operator()(Args... args) {
            static_assert(sizeof...(args) == N, "Wrong number of arguments passed");
            std::array<std::size_t, N> idx_array = { static_cast<std::size_t>(args)... };
            for (size_t i = 0; i < N; ++i) {
                if (idx_array[i] >= dimensions_[i]) {
                    std::ostringstream oss;
                    oss << "Index out of bounds for dimension " << i
                        << ": " << idx_array[i] << " >= " << dimensions_[i];
                    throw std::out_of_range(oss.str());
                }
            }
            return data_[calculate_linear_index(idx_array)];
        }

        template<typename... Args>
        void reshape(Args&&... args) {
            std::vector<std::size_t> new_dims = { static_cast<std::size_t>(std::forward<Args>(args))... };
            validate_dimensions(new_dims);
            size_t new_size = 1;
            for (const auto& dim : new_dims) {
                new_size *= dim;
            }
            if (new_size > data_.size()) {
                data_.resize(new_size, T());
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

        std::array<std::size_t, N> shape() const {
            return dimensions_;
        }

        void fill(const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        iterator begin() { return data_.begin(); }
        iterator end() { return data_.end(); }
        const_iterator begin() const { return data_.begin(); }
        const_iterator end() const { return data_.end(); }
        const_iterator cbegin() const { return data_.cbegin(); }
        const_iterator cend() const { return data_.cend(); }

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T, N>& tensor) {
            if (tensor.data_.empty()) {
                os << "{}";
                return os;
            }
            std::array<std::size_t, N> indices{};
            tensor.print_tensor_recursive(os, indices, 0);
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

        // esto es tensor-tensor

        Tensor<T, N> operator+(const Tensor<T, N>& other) const {
            if (!are_broadcastable(dimensions_, other.dimensions_)) {
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            auto result_shape = get_broadcast_result_shape(dimensions_, other.dimensions_);
            Tensor<T, N> result(result_shape);

            std::array<std::size_t, N> idx{};
            const size_t total = std::accumulate(result_shape.begin(), result_shape.end(), (size_t)1, std::multiplies<size_t>());

            for (size_t linear = 0; linear < total; ++linear) {
                // Convierte índice lineal a multidimensional
                size_t remainder = linear;
                for (int dim = N - 1; dim >= 0; --dim) {
                    idx[dim] = remainder % result_shape[dim];
                    remainder /= result_shape[dim];
                }

                auto idx1 = get_broadcasted_index(idx, dimensions_);
                auto idx2 = get_broadcasted_index(idx, other.dimensions_);

                result.data_[linear] = data_[calculate_linear_index(idx1)] + other.data_[other.calculate_linear_index(idx2)];
            }
            return result;
        }

        Tensor<T, N> operator-(const Tensor<T, N>& other) const {
            if (!are_broadcastable(dimensions_, other.dimensions_)) {
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            auto result_shape = get_broadcast_result_shape(dimensions_, other.dimensions_);
            Tensor<T, N> result(result_shape);

            std::array<std::size_t, N> idx{};
            const size_t total = std::accumulate(result_shape.begin(), result_shape.end(), (size_t)1, std::multiplies<size_t>());

            for (size_t linear = 0; linear < total; ++linear) {
                // Convierte índice lineal a multidimensional
                size_t remainder = linear;
                for (int dim = N - 1; dim >= 0; --dim) {
                    idx[dim] = remainder % result_shape[dim];
                    remainder /= result_shape[dim];
                }

                auto idx1 = get_broadcasted_index(idx, dimensions_);
                auto idx2 = get_broadcasted_index(idx, other.dimensions_);

                result.data_[linear] = data_[calculate_linear_index(idx1)] - other.data_[other.calculate_linear_index(idx2)];
            }
            return result;
        }

        Tensor<T, N> operator*(const Tensor<T, N>& other) const {
            if (!are_broadcastable(dimensions_, other.dimensions_)) {
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            auto result_shape = get_broadcast_result_shape(dimensions_, other.dimensions_);
            Tensor<T, N> result(result_shape);

            std::array<std::size_t, N> idx{};
            const size_t total = std::accumulate(result_shape.begin(), result_shape.end(), (size_t)1, std::multiplies<size_t>());

            for (size_t linear = 0; linear < total; ++linear) {
                size_t remainder = linear;
                for (int dim = N - 1; dim >= 0; --dim) {
                    idx[dim] = remainder % result_shape[dim];
                    remainder /= result_shape[dim];
                }

                auto idx1 = get_broadcasted_index(idx, dimensions_);
                auto idx2 = get_broadcasted_index(idx, other.dimensions_);

                result.data_[linear] = data_[calculate_linear_index(idx1)] * other.data_[other.calculate_linear_index(idx2)];
            }
            return result;
        }

        Tensor<T, N> operator/(const Tensor<T, N>& other) const {
            if (!are_broadcastable(dimensions_, other.dimensions_)) {
                throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            auto result_shape = get_broadcast_result_shape(dimensions_, other.dimensions_);
            Tensor<T, N> result(result_shape);

            std::array<std::size_t, N> idx{};
            const size_t total = std::accumulate(result_shape.begin(), result_shape.end(), (size_t)1, std::multiplies<size_t>());

            for (size_t linear = 0; linear < total; ++linear) {
                size_t remainder = linear;
                for (int dim = N - 1; dim >= 0; --dim) {
                    idx[dim] = remainder % result_shape[dim];
                    remainder /= result_shape[dim];
                }

                // Obtiene índices broadcasted para cada tensor
                auto idx1 = get_broadcasted_index(idx, dimensions_);
                auto idx2 = get_broadcasted_index(idx, other.dimensions_);

                if (other.data_[other.calculate_linear_index(idx2)] == T{}) {
                    throw std::domain_error("Division by zero");
                }
                result.data_[linear] = data_[calculate_linear_index(idx1)] / other.data_[other.calculate_linear_index(idx2)];
            }
            return result;
        }

        Tensor<T, N> operator+(const T& scalar) const {
            Tensor<T, N> result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] += scalar;
            }
            return result;
        }

        Tensor<T, N> operator-(const T& scalar) const {
            Tensor<T, N> result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] -= scalar;
            }
            return result;
        }

        Tensor<T, N> operator*(const T& scalar) const {
            Tensor<T, N> result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] *= scalar;
            }
            return result;
        }

        Tensor<T, N> operator/(const T& scalar) const {
            if (scalar == T{}) {
                throw std::domain_error("Division by zero");
            }
            Tensor<T, N> result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] /= scalar;
            }
            return result;
        }

        friend Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
            return tensor + scalar;
        }

        friend Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& tensor) {
            return tensor * scalar;
        }

        // Note: scalar - tensor and scalar / tensor are not commutative, so they need different implementations
        friend Tensor<T, N> operator-(const T& scalar, const Tensor<T, N>& tensor) {
            Tensor<T, N> result(tensor.dimensions_);
            for (size_t i = 0; i < tensor.data_.size(); ++i) {
                result.data_[i] = scalar - tensor.data_[i];
            }
            return result;
        }

        friend Tensor<T, N> operator/(const T& scalar, const Tensor<T, N>& tensor) {
            Tensor<T, N> result(tensor.dimensions_);
            for (size_t i = 0; i < tensor.data_.size(); ++i) {
                if (tensor.data_[i] == T{}) {
                    throw std::domain_error("Division by zero");
                }
                result.data_[i] = scalar / tensor.data_[i];
            }
            return result;
        }


        Tensor<T, N> transpose_2d() const {
            if constexpr (N < 2) {
                throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
            }

            std::array<std::size_t, N> new_shape = dimensions_;
            std::swap(new_shape[N-2], new_shape[N-1]);

            Tensor<T, N> result(new_shape);

            // Para cada elemento, calcular los índices transpuestos
            std::array<std::size_t, N> indices{};
            const size_t total = calculate_total_elements();

            for (size_t linear = 0; linear < total; ++linear) {
                // Convertir índice lineal a multidimensional usando las dimensiones originales
                size_t remainder = linear;
                for (int dim = N - 1; dim >= 0; --dim) {
                    indices[dim] = remainder % dimensions_[dim];
                    remainder /= dimensions_[dim];
                }

                // crear índices transpuestos: intercambiar las dos últimas dimensiones
                std::array<std::size_t, N> transposed_indices = indices;
                std::swap(transposed_indices[N-2], transposed_indices[N-1]);

                // se calcul a el índice lineal en el tensor resultado
                size_t result_linear = 0;
                size_t stride = 1;
                for (int dim = N - 1; dim >= 0; --dim) {
                    result_linear += transposed_indices[dim] * stride;
                    stride *= new_shape[dim];
                }

                result.data_[result_linear] = data_[linear];
            }

            return result;
        }

        size_t get_linear_index(const std::array<std::size_t, N>& indices) const {
            return calculate_linear_index(indices);
        }

        template<typename U, std::size_t M>
        friend Tensor<U, M> matrix_product(const Tensor<U, M>& tensor1, const Tensor<U, M>& tensor2);
    };

    template<typename T, std::size_t N>
    Tensor<T, N> transpose_2d(const Tensor<T, N>& tensor) {
        return tensor.transpose_2d();
    }

    template<typename T, std::size_t N>
    Tensor<T, N> matrix_product(const Tensor<T, N>& tensor1, const Tensor<T, N>& tensor2) {
        static_assert(N >= 2, "Tensors must have at least 2 dimensions for matrix multiplication");

        auto shape1 = tensor1.shape();
        auto shape2 = tensor2.shape();

        // verificar compatibilidad de las dimensiones de matriz (las dos últimas)
        std::size_t rows1 = shape1[N-2];      // filas del primer tensor
        std::size_t cols1 = shape1[N-1];      // columnas del primer tensor
        std::size_t rows2 = shape2[N-2];      // filas del segundo tensor
        std::size_t cols2 = shape2[N-1];      // columnas del segundo tensor

        bool matrix_compatible = (cols1 == rows2);

        bool batch_compatible = true;
        for (std::size_t i = 0; i < N - 2; ++i) {
            if (shape1[i] != shape2[i]) {
                batch_compatible = false;
                break;
            }
        }

        if (!matrix_compatible && !batch_compatible) {
            throw std::invalid_argument("Matrix dimensions are not compatible for multiplication AND Batch dimensions do not match");
        }
        else if (!matrix_compatible) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        }
        else if (!batch_compatible) {
            throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }

        std::array<std::size_t, N> result_shape = shape1;
        result_shape[N-2] = rows1;  // filas del resultado = filas del primer tensor
        result_shape[N-1] = cols2;  // columnas del resultado = columnas del segundo tensor

        Tensor<T, N> result(result_shape);
        result.fill(T{});  // Inicializar con ceros

        // Realizar la multiplicación de matrices
        // Para cada batch
        std::array<std::size_t, N-2> batch_indices{};
        std::size_t total_batches = 1;
        for (std::size_t i = 0; i < N - 2; ++i) {
            total_batches *= shape1[i];
        }

        for (std::size_t batch = 0; batch < total_batches; ++batch) {
            // Convertir índice de batch lineal a multidimensional
            std::size_t remainder = batch;
            for (int dim = N - 3; dim >= 0; --dim) {
                batch_indices[dim] = remainder % shape1[dim];
                remainder /= shape1[dim];
            }

            for (std::size_t i = 0; i < rows1; ++i) {
                for (std::size_t j = 0; j < cols2; ++j) {
                    T sum = T{};
                    for (std::size_t k = 0; k < cols1; ++k) {
                        std::array<std::size_t, N> idx1;
                        for (std::size_t d = 0; d < N - 2; ++d) {
                            idx1[d] = batch_indices[d];
                        }
                        idx1[N-2] = i;
                        idx1[N-1] = k;

                        std::array<std::size_t, N> idx2;
                        for (std::size_t d = 0; d < N - 2; ++d) {
                            idx2[d] = batch_indices[d];
                        }
                        idx2[N-2] = k;
                        idx2[N-1] = j;

                        sum += tensor1.data_[tensor1.get_linear_index(idx1)] *
                               tensor2.data_[tensor2.get_linear_index(idx2)];
                    }

                    std::array<std::size_t, N> result_idx;
                    for (std::size_t d = 0; d < N - 2; ++d) {
                        result_idx[d] = batch_indices[d];
                    }
                    result_idx[N-2] = i;
                    result_idx[N-1] = j;

                    result.data_[result.get_linear_index(result_idx)] = sum;
                }
            }
        }

        return result;
    }
}

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H