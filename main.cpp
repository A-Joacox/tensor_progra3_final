#include <iostream>
#include "src/tensor.h"

using std::exception;

int main() {
    try {
        const utec::algebra::Tensor<int, 2> t(2, 2, 2);
    }
    catch (const exception& e) {
        std::cout << e.what();  // Mensaje de ERROR: Number of dimensions do not match with 2
    }
    return 0;
}
