#include <iostream>

#include "error.h"
#include "tensor.h"

namespace rnnpp {

std::ostream& operator<<(std::ostream &os, const Tensor &t) {
  RNNPP_CHECK(t.dim.shape.size() == 2, "Tensor must be two dimension");
  for (int i=0; i < t.dim[0]; ++i) {
    int offset = i * t.dim[1];
    for (int j=0; j < t.dim[1]-1; ++j) {
      os << t.data[offset + j] << ",";
    }
    os << t.data[offset + t.dim[1]-1] << std::endl;
  }
//  os << std::endl;
  return os;
}

Tensor Tensor::transpose() {
  Tensor dest;

  std::vector<int> d(dim.shape.size());

  // Example: [2, 1, 5, 3, 4] ==>  [4, 3, 5, 1, 2]
  for (int i=0; i < dim.shape.size() / 2 + dim.shape.size() % 2; ++i) {
    d[i] = dim[dim.shape.size() - 1 - i];
    d[dim.shape.size() - 1 - i] = dim[i];
  }
  dest.dim = Dim(d, dim.batch_size);

  dest.data = data;
  return dest;
}

void elementwise_add(const Tensor &src, Tensor &dest) {
  for (int i=0; i < dest.dim.size(); ++i) {
    dest.data[i] += src.data[i];
  }
}

void elementwise_subtract(const Tensor &src, Tensor &dest) {
  for (int i=0; i < dest.dim.size(); ++i) {
    dest.data[i] -= src.data[i];
  }
}


// (M, N) = (M, K) x (K, N)
void matmul(const Tensor &lhs, const Tensor &rhs, Tensor &dest) {
  int M = dest.dim[0];
  int N = dest.dim[1];
  int K = lhs.dim[1];
  int max_b = std::max(lhs.dim.batch_size, rhs.dim.batch_size);

//  std::cout << dest.dim << "=" << lhs.dim << " x " << rhs.dim << std::endl;

  RNNPP_CHECK(lhs.dim[1] == rhs.dim[0], "Invalid dimension in matmul");
  RNNPP_CHECK(dest.dim[0] == lhs.dim[0], "Invalid dimension in matmul");
  RNNPP_CHECK(dest.dim[1] == rhs.dim[1], "Invalid dimension in matmul");

  const float* ld = lhs.cdata();
  const float* rd = rhs.cdata();
  float* destd = dest.data;

  for (int i=0; i < dest.dim.size(); ++i) {
    destd[i] = 0.;
  }

  for (int b=0; b < max_b; ++b) {
    for (int m=0; m < M; ++m) {
      int offset1 = m * N;
      int offset2 = m * K;
      for (int n=0; n < N; ++n) {
        for (int k=0; k < K; ++k) {
          int offset3 = k * N;

//          float l = ld[lhs.dim.size() * b % lhs.dim.batch_size + offset2 + k];
//          float r = rd[rhs.dim.size() * b % rhs.dim.batch_size + offset3 + n];
//          std::cout << "["<<m<<","<<n<<"]"<< " += [" <<m<<","<<k<<"]" << l << " * "\
//                    << "["<<k<<","<<n<<"]"<< r << std::endl;

          destd[offset1 + n] += ld[lhs.dim.size() * b % lhs.dim.batch_size + offset2 + k] \
                                * rd[rhs.dim.size() * b % rhs.dim.batch_size + offset3 + n];
        }
      }
    }
  }
}

} // namespace rnnpp
