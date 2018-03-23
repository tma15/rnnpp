#ifndef RNNPP_PARAMETER_H_
#define RNNPP_PARAMETER_H_

#include "dim.h"
#include "tensor.h"

namespace rnnpp {

class Initializer {
  public:
    Initializer() {}
    ~Initializer() {}

    void init(Tensor &t);
};

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

class LookupParameter {
  public:
    LookupParameter() {}

    LookupParameter(const Dim &dim);

    ~LookupParameter() {}

    Tensor all_values;

    std::vector<Tensor> values;
    std::vector<Tensor> grads;
};

} // namespace rnnpp

#endif // RNNPP_PARAMETER_H_
