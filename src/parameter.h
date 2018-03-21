#ifndef RNNPP_PARAMETER_H_
#define RNNPP_PARAMETER_H_

#include "dim.h"
#include "tensor.h"

namespace rnnpp {

class Parameter {
  public:
    Parameter() {}

    Parameter(const Dim &dim) {
      value.data = new float[dim.size()];
      value.dim = dim;
      grad.data = new float[dim.size()];
      grad.dim = dim;
    }

    ~Parameter() {}

    Tensor value;
    Tensor grad;
};

} // namespace rnnpp

#endif // RNNPP_PARAMETER_H_
