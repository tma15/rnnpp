#ifndef RNNPP_TENSOR_H_
#define RNNPP_TENSOR_H_

#include "dim.h"

namespace rnnpp {

class Tensor {
  public:
    Tensor() {}

    Tensor(const Dim &d, const std::vector<float> &v): dim(d) {
      std::memcpy(data, v.data(), sizeof(float) * v.size());
    }

    ~Tensor() {}

    friend std::ostream& operator<<(std::ostream &os, const Tensor &t);

    float* cdata() const { return data; }

    float* rdata() { return data; }

    Tensor transpose();

    float *data;
    Dim dim;
};

void elementwise_add(const Tensor &src, Tensor &dest);

void matmul(const Tensor &lhs, const Tensor &rhs, Tensor &dest);

} // namespace rnnpp



#endif // RNNPP_TESOR_H_
