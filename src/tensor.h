#ifndef RNNPP_TENSOR_H_
#define RNNPP_TENSOR_H_

#include <Eigen/Core>

#include "dim.h"

namespace rnnpp {

//typedef Eigen::MatrixXf Tensor;

class Tensor {
  public:
    Tensor() {}
    ~Tensor() {}

    float *data;
    Dim dim;
};

} // namespace rnnpp



#endif // RNNPP_TESOR_H_
