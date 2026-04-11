#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <string>
#include <cassert>

int main() {
    std::cout << "Testing owTensor with std::string..." << std::endl;

    // 1. Create a 1D string tensor
    ow::owTensor<std::string, 1> s1(3);
    s1(0) = "Hello";
    s1(1) = "OpenWhiz";
    s1(2) = "Unification";

    std::cout << "1D Tensor content:" << std::endl;
    s1.print();

    assert(s1(0) == "Hello");
    assert(s1(1) == "OpenWhiz");
    assert(s1(2) == "Unification");

    // 2. Create a 2D string tensor
    ow::owTensor<std::string, 2> s2(2, 2);
    s2(0, 0) = "A";
    s2(0, 1) = "B";
    s2(1, 0) = "C";
    s2(1, 1) = "D";

    std::cout << "\n2D Tensor content:" << std::endl;
    s2.print();

    assert(s2(0, 0) == "A");
    assert(s2(1, 1) == "D");

    // 3. Test copy constructor
    ow::owTensor<std::string, 1> s3 = s1;
    assert(s3(1) == "OpenWhiz");
    s3(1) = "Modified";
    assert(s1(1) == "OpenWhiz"); // Deep copy check
    assert(s3(1) == "Modified");

    std::cout << "\nAll string tensor tests passed!" << std::endl;

    return 0;
}
