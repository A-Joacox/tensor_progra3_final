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
                }return linear_idx;
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

        public:

            using iterator = typename std::vector<T>::iterator;
            using const_iterator = typename std::vector<T>::const_iterator;

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

            // Versión mutable (devuelve T&)
            template<typename... Args>
            T& operator()(Args... args) {
                static_assert(sizeof...(args) == N, "Wrong number of arguments passed");
                std::array<size_t, N> idx_array = { static_cast<size_t>(args)... };
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

            // Versión const (devuelve const T&)
            template<typename... Args>
            const T& operator()(Args... args) const {
                static_assert(sizeof...(args) == N, "Wrong number of arguments passed");
                std::array<size_t, N> idx_array = { static_cast<size_t>(args)... };
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


            static std::array<std::size_t, N> broadcast_shape(const std::array<std::size_t, N>& A, const std::array<std::size_t, N>& B)
            {
                std::array<std::size_t, N> R;
                for (size_t d = 0; d < N; ++d) {
                    auto a = A[d], b = B[d];
                    if (a == b) {
                        R[d] = a;
                    } else if (a == 1) {
                        R[d] = b;
                    } else if (b == 1) {
                        R[d] = a;
                    } else {
                        throw std::invalid_argument(
                           "Shapes do not match and they are not compatible for broadcasting");
                    }
                }
                return R;
            }

            Tensor<T, N> operator+(const Tensor<T, N>& other) const {
                auto shapeA = dimensions_;
                auto shapeB = other.dimensions_;
                auto shapeR = broadcast_shape(shapeA, shapeB);
                Tensor<T, N> result;
                result.dimensions_ = shapeR;
                size_t totalR = 1;
                for (auto x : shapeR)
                    totalR *= x;
                result.data_.resize(totalR);

                for (size_t linear = 0; linear < totalR; ++linear) {
                    size_t rem = linear;
                    std::array<std::size_t, N> idxR;
                        for (size_t d = 0; d < N; ++d) {
                            size_t stride = 1;
                            for (size_t e = d + 1; e < N; ++e)
                                stride *= shapeR[e];
                                idxR[d] = rem / stride;
                                rem %= stride;
                            }
                     std::array<std::size_t, N> idxA, idxB;
                         for (size_t d = 0; d < N; ++d) {
                            idxA[d] = (shapeA[d] == 1 ? 0 : idxR[d]);
                            idxB[d] = (shapeB[d] == 1 ? 0 : idxR[d]);
                            }
                    size_t offA = 0, offB = 0;
                    size_t mulA = 1, mulB = 1;
                    for (size_t d = N; d-- > 0; ) {
                        offA += idxA[d] * mulA;
                        mulA *= shapeA[d];
                        offB += idxB[d] * mulB;
                        mulB *= shapeB[d];
                        }
                result.data_[linear] = data_[offA] + other.data_[offB];
            }
                return result;
}

Tensor<T, N> operator-(const Tensor<T, N>& other) const {
    auto shapeA = dimensions_;
    auto shapeB = other.dimensions_;
    auto shapeR = broadcast_shape(shapeA, shapeB);
    Tensor<T, N> result;
    result.dimensions_ = shapeR;
    size_t totalR = 1;
    for (auto x : shapeR) totalR *= x;
    result.data_.resize(totalR);

    for (size_t linear = 0; linear < totalR; ++linear) {
        size_t rem = linear;
        std::array<std::size_t, N> idxR;
        for (size_t d = 0; d < N; ++d) {
            size_t stride = 1;
            for (size_t e = d + 1; e < N; ++e)
                stride *= shapeR[e];
            idxR[d] = rem / stride;
            rem %= stride;
        }
        std::array<std::size_t, N> idxA, idxB;
        for (size_t d = 0; d < N; ++d) {
            idxA[d] = (shapeA[d] == 1 ? 0 : idxR[d]);
            idxB[d] = (shapeB[d] == 1 ? 0 : idxR[d]);
        }
        size_t offA = 0, offB = 0;
        size_t mulA = 1, mulB = 1;
        for (size_t d = N; d-- > 0; ) {
            offA += idxA[d] * mulA;
            mulA *= shapeA[d];
            offB += idxB[d] * mulB;
            mulB *= shapeB[d];
        }
        result.data_[linear] = data_[offA] - other.data_[offB];
    }
    return result;
}

            Tensor<T, N> operator-(const T number) const {
                Tensor<T, N> result(*this);
                for (size_t i = 0; i < data_.size(); ++i) {
                    result.data_[i] -= number;
                }
                return result;
            }

            Tensor<T, N> operator+(const T number) const {
                Tensor<T, N> result(*this);
                for (size_t i = 0; i < data_.size(); ++i) {
                    result.data_[i] += number;
                }
                return result;
            }

            Tensor<T, N> operator/(const T number) const {
                Tensor<T, N> result(*this);
                for (size_t i = 0; i < data_.size(); ++i) {
                    result.data_[i] = result.data_[i]/number;
                }
                return result;
            }


            Tensor<T, N> operator*(const Tensor<T, N>& other) const {
                auto shapeA = dimensions_;
                auto shapeB = other.dimensions_;
                auto shapeR = broadcast_shape(shapeA, shapeB);
                Tensor<T, N> result;
                result.dimensions_ = shapeR;
                size_t totalR = 1;
                for (auto x : shapeR) totalR *= x;
                result.data_.resize(totalR);

                for (size_t linear = 0; linear < totalR; ++linear) {
                    size_t rem = linear;
                    std::array<std::size_t, N> idxR;
                    for (size_t d = 0; d < N; ++d) {
                        size_t stride = 1;
                        for (size_t e = d + 1; e < N; ++e)
                            stride *= shapeR[e];
                        idxR[d] = rem / stride;
                        rem %= stride;
                    }
                    std::array<std::size_t, N> idxA, idxB;
                    for (size_t d = 0; d < N; ++d) {
                        idxA[d] = (shapeA[d] == 1 ? 0 : idxR[d]);
                        idxB[d] = (shapeB[d] == 1 ? 0 : idxR[d]);
                    }
                    size_t offA = 0, offB = 0;
                    size_t mulA = 1, mulB = 1;
                    for (size_t d = N; d-- > 0; ) {
                        offA += idxA[d] * mulA;
                        mulA *= shapeA[d];
                        offB += idxB[d] * mulB;
                        mulB *= shapeB[d];
                    }
                    result.data_[linear] = data_[offA] * other.data_[offB];
                }
                return result;
            }


        };

        template<typename T, size_t N>
        Tensor<T, N> transpose_2d(const Tensor<T, N>& A) {
            if constexpr (N < 2) {
                throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");
            }

            auto shapeA = A.shape(); // std::array<size_t, N>

            std::array<size_t, N> shapeR = shapeA;
            std::swap(shapeR[N - 1], shapeR[N - 2]);

            auto make_B = [&]<size_t... Is>(std::index_sequence<Is...>) {
                return Tensor<T, N>(shapeR[Is]...);
            };
            Tensor<T, N> B = make_B(std::make_index_sequence<N>{});

            size_t total = 1;
            for (size_t d = 0; d < N; ++d) {
                total *= shapeR[d];
            }

            for (size_t lin = 0; lin < total; ++lin) {
                size_t rem = lin;
                std::array<size_t, N> idxA;
                for (size_t d = 0; d < N; ++d) {
                    size_t stride = 1;
                    for (size_t e = d + 1; e < N; ++e) {
                        stride *= shapeA[e];
                    }
                    idxA[d] = rem / stride;
                    rem %= stride;
                }

                std::array<size_t, N> idxR = idxA;
                std::swap(idxR[N - 1], idxR[N - 2]);

                size_t linR = 0;
                size_t mul = 1;
                for (size_t d = N; d-- > 0; ) {
                    linR += idxR[d] * mul;
                    mul *= shapeR[d];
                }

                B[linR] = A[lin];
            }

            return B;
        }

        template<typename T, size_t N>
        Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& A) {
            return A + scalar;
        }

        template<typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& A, const Tensor<T, N>& B) {
            if constexpr (N != 3) {
                throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
            } else {
                auto shapeA = A.shape(); // {batchA, M, K}
                auto shapeB = B.shape(); // {batchB, Kp, Ncol}
                size_t batchA = shapeA[0], M = shapeA[1], K = shapeA[2];
                size_t batchB = shapeB[0], Kp = shapeB[1], Ncol = shapeB[2];

                if (K != Kp) {
                    throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
                }
                if (batchA != batchB) {
                    throw std::runtime_error(
                        "Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
                }

                Tensor<T, 3> C(batchA, M, Ncol);
                C.fill(T{});

                for (size_t i = 0; i < batchA; ++i) {
                    for (size_t p = 0; p < M; ++p) {
                        for (size_t q = 0; q < Ncol; ++q) {
                            T acc = T{};
                            for (size_t r = 0; r < K; ++r) {
                                acc += A(i, p, r) * B(i, r, q);
                            }
                            C(i, p, q) = acc;
                        }
                    }
                }
                return C;
            }
        }


    }

    #endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

