#include <iostream>
#include </home/kdot/cpp_libraries/eigen-3.3.9/Eigen/Dense>

using namespace Eigen;

int main() {
    // Dynamic- Resizable
    MatrixXd m;

    // Fixed Sized Matrix
    Matrix3d f;

    f = Matrix3d::Constant(1.0);

    std::cout << f <<  std::endl;



}



