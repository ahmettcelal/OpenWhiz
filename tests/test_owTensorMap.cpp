#include <iostream>
#include <cassert>
#include <vector>
#include "OpenWhiz/owTensor.hpp"

int main() {
    std::cout << "Testing owTensorMap..." << std::endl;

    // 1. Create a raw memory region (std::vector as buffer)
    std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};

    // 2. Map this buffer to a 2x2 owTensorMap
    ow::owTensor<float, 2>::owTensorShape shape = {2, 2};
    ow::owTensorMap<float, 2> tensorMap(buffer.data(), shape);

    std::cout << "Mapped Tensor Content:" << std::endl;
    tensorMap.print();

    // 3. Verify access
    assert(tensorMap(0, 0) == 1.0f);
    assert(tensorMap(1, 1) == 4.0f);

    // 4. Modify via TensorMap and check raw buffer
    tensorMap(0, 1) = 10.0f;
    std::cout << "\nAfter modification via map (tensorMap(0,1) = 10.0):" << std::endl;
    std::cout << "Raw Buffer[1]: " << buffer[1] << " (Expected: 10.0)" << std::endl;
    
    assert(buffer[1] == 10.0f);

    // 5. Test mathematical operations (Copying from map to new tensor)
    ow::owTensor<float, 2> mat2({2, 2}, 1.0f);
    auto sum = tensorMap + mat2; // Should result in [[2, 11], [4, 5]]
    
    std::cout << "\nSum (Map + owTensor Ones):" << std::endl;
    sum.print();
    assert(sum(0, 1) == 11.0f);

    std::cout << "\nowTensorMap test passed!" << std::endl;

    return 0;
}
