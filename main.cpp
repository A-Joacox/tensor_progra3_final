#include <iostream>
#include "src/tensor.h"

using std::exception;

int main() {
    utec::algebra::Tensor<double, 2> t1(5, 3);
    utec::algebra::Tensor<double, 2> t2(5, 2);
    t1.fill(10);
    t2.fill(5);
    try {
        const auto r = utec::algebra::matrix_product(t1, t2);
    }
    catch (const exception& e) {
        std::cout << e.what(); // Mensaje de ERROR: Matrix dimensions are incompatible for multiplication
    }

}
