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
                }return linear_idx;
            }

            template <typename OStream>
            void print_tensor_recursive(OStream& os, std::array<std::size_t, N> indices,
                                       size_t dim_idx) const {
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

        public:
            Tensor() : dimensions_{}, data_() {}

            //Constructor con dimensiones especificas
            template<typename... Args>
            Tensor(Args&&... args) {
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
        };
    }

    #endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

