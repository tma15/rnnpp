#ifndef RNNPP_PARAMETER_H_
#define RNNPP_PARAMETER_H_

#include "dim.h"
#include "tensor.h"

namespace rnnpp {

class Parameter {
  public:
    Parameter() {}

    Parameter(const Dim &dim) {
      data_.data = new float[dim.size()];
      data_.dim = dim;
      grad_.data = new float[dim.size()];
      grad_.dim = dim;
    }

    ~Parameter() {}

    Tensor data_;
    Tensor grad_;
};

} // namespace rnnpp

#endif // RNNPP_PARAMETER_H_
