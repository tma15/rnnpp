#ifndef RNNPP_TENSOR_H_
#define RNNPP_TENSOR_H_

#include "dim.h"

namespace rnnpp {

namespace internal {

constexpr int adder(const std::vector<int> &shape, int k, int arg) {
  return shape[k] * arg;
}

template<typename ... Args> 
constexpr int adder(const std::vector<int> &shape, int k, int head, Args... tail) {
  return shape[k] * head + adder(shape, k + 1, tail...);
}

} // internal

class Tensor {
  public:
    Tensor(): dim(Dim()), data(nullptr) {}

    Tensor(const Dim &d, const std::vector<float> &v): dim(d) {
      data = new float[v.size()];
      std::memcpy(data, v.data(), sizeof(float) * v.size());
    }

    ~Tensor() {}

    friend std::ostream& operator<<(std::ostream &os, const Tensor &t);

    template <typename ... Args>
    float& operator() (Args const & ... args) {
      int k = internal::adder(dim.stride, 0, args...);
      return data[k];
    }


    float* cdata() const { return data; }

    float* rdata() { return data; }

    Tensor transpose();

    float *data;
    Dim dim;
};

void assign(const Tensor &src, Tensor &dest);

void elementwise_add(const Tensor &src, Tensor &dest);

void elementwise_sub(const Tensor &src, Tensor &dest);

void elementwise_square(Tensor &dest);

void matmul(const Tensor &lhs, const Tensor &rhs, Tensor &dest);

} // namespace rnnpp



#endif // RNNPP_TESOR_H_
